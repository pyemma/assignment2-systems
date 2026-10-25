import torch
import torch.distributed as dist

from cs336_basics.model import Linear, Embedding


class FSDP(torch.nn.Module):

    @staticmethod
    def _all_gather_param(module, world_size, compute_dtype, skip_master=False):
        # keep a copy of master weights, which needs to be assigned back, the comm and compute all done in another dtype
        if not skip_master:
            module._saved_fp32 = module.weight.data
            module.weight.data = module.weight.data.to(compute_dtype) if compute_dtype else module.weight.data
        shape = (world_size * module.weight.shape[0],) + module.weight.shape[1:]
        param = torch.empty(shape, dtype=module.weight.dtype, device=module.weight.device)
        handle = dist.all_gather_into_tensor(param.data, module.weight.data, async_op=True)
        return param, handle

    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.module = module
        self.compute_dtype = compute_dtype

        self._index_to_module = {}  # used to quickly find the next module to unshard
        self._handles = []

        def _unshard(module, input):
            """
                All gather the parameters in the module, assuming Linear and Embedding type of module for now
            """
            idx = module.idx
            if idx == 0:  # this is the cold start part, immediately start the all gather for the first 2 layers
                buffer, handle = FSDP._all_gather_param(module, self.world_size, self.compute_dtype)
                next_module = self._index_to_module[1]
                buffer1, handle1 = FSDP._all_gather_param(next_module, self.world_size, self.compute_dtype)
                next_module.buffer = buffer1
                next_module.handle = handle1
            else:
                buffer, handle = module.buffer, module.handle
                del module.buffer
                del module.handle

            handle.wait()
            module.weight.data = buffer.data
            return input

        def _init_shard(module):
            # this only works for the test, in reality there could be uneven size of
            # the input tensor and padding would be required
            n = module.weight.shape[0] // self.world_size
            module.weight.data = module.weight.data[self.rank*n:(self.rank+1)*n].clone()

        def _shard(module, input, output):
            """
                Shard the parameters in the module back to local rank slice
            """
            assert type(module) == Linear or type(module) == Embedding
            idx = module.idx
            module.weight.data = module._saved_fp32
            del module._saved_fp32

            if idx + 2 < len(self._index_to_module):
                # start the fetch of next layer
                next_module = self._index_to_module[idx+2]
                next_buffer, next_handle = FSDP._all_gather_param(next_module, self.world_size, self.compute_dtype)
                next_module.buffer = next_buffer
                next_module.handle = next_handle
            return output

        def _unshard_bwd(module, grad_output):
            """
                Different from the forward that there is no overlapping
            """
            idx = module.idx
            if idx == len(self._index_to_module) - 1:
                buffer, handle = FSDP._all_gather_param(module, self.world_size, self.compute_dtype)
                module1 = self._index_to_module[idx - 1]
                buffer1, handle1 = FSDP._all_gather_param(module1, self.world_size, self.compute_dtype)
                module1.buffer = buffer1
                module1.handle = handle1
            else:
                buffer, handle = module.buffer, module.handle
                del module.buffer
                del module.handle

            handle.wait()
            module.weight.data = buffer.data
            return grad_output

        def make_param_hook(module, rank, world_size):
            def hook(param):
                if type(module) == Linear or type(module) == Embedding:
                    n = param.grad.shape[0] // world_size
                    output_tensor = torch.empty((n,) + param.shape[1:], dtype=param.grad.dtype, device=param.grad.device)
                    h = dist.reduce_scatter_tensor(output_tensor.data, param.grad.data, op=dist.ReduceOp.AVG, async_op=True)
                    param.data = module._saved_fp32
                    del module._saved_fp32
                    self._handles.append((h, param, output_tensor))

                    idx = module.idx
                    if idx - 2 >= 0:
                        prev_module = self._index_to_module[idx - 2]
                        prev_buffer, prev_handle = FSDP._all_gather_param(prev_module, self.world_size, self.compute_dtype)
                        prev_module.buffer = prev_buffer
                        prev_module.handle = prev_handle
                else:
                    h = dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG, async_op=True)
                    self._handles.append((h, param, None))

            return hook
        
        idx = 0
        for m in self.module.modules():
            if type(m) == Linear or type(m) == Embedding:
                m.idx = idx
                self._index_to_module[idx] = m
                _init_shard(m)
                m.register_forward_pre_hook(_unshard)
                m.register_forward_hook(_shard)
                m.register_full_backward_pre_hook(_unshard_bwd)
                idx += 1
            
            for p in m.parameters(recurse=False):
                if p.requires_grad:
                    p.register_post_accumulate_grad_hook(make_param_hook(m, self.rank, self.world_size))

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        """
            This is not the optimal solution as the gradient waits until finish to be freed
        """
        for h, p, o in self._handles:
            h.wait()
            if o is None:
                continue
            p.grad = o.to(torch.float32)
        self._handles.clear()

    def fsdp_gather_full_params(self):
        dic = {}
        named_modules = dict(self.module.named_modules())
        for name, param in self.module.named_parameters():
            if not param.requires_grad:
                continue
            parts = name.rsplit(".", 1)
            module = named_modules[parts[0]]
            if type(module) == Linear or type(module) == Embedding:
                buffer, handle = FSDP._all_gather_param(module, self.world_size, self.compute_dtype, skip_master=True)
                handle.wait()
                dic[name] = buffer.data
            else:
                dic[name] = param.data
        return dic