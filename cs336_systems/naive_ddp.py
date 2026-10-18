import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


class DDP(torch.nn.Module):
    
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.module = model
        self.global_rank = dist.get_rank()
        self._handles = []
        
        # broadcast the rank
        for t in self.module.state_dict().values():
            dist.broadcast(t, src=0)

        # register the backward hook
        def hook(p: torch.Tensor) -> None:
            h = dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True)
            self._handles.append(h)
        
        trainable_params = [p for p in self.module.parameters() if p.requires_grad]
        for p in trainable_params:
            p.register_post_accumulate_grad_hook(lambda p: hook(p))

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for h in self._handles:
            h.wait()
        self._handles.clear()


def gradient_synchronization(model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    # should category by dtype probably
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    grads = [p.grad for p in trainable_params]
    flatten_buffer = _flatten_dense_tensors(grads)
    dist.all_reduce(flatten_buffer, op=dist.ReduceOp.AVG, async_op=False)
    reduced_grads = _unflatten_dense_tensors(flatten_buffer, grads)
    for g, p in zip(reduced_grads, trainable_params):
        p.grad = g 
