from typing import Any, Type

import math

import torch
import torch.distributed as dist
import torch.optim as optim


class ShardedOptimizer(optim.Optimizer):

    def __init__(self, params, optimizer_cls: Type[optim.Optimizer], **kwargs: Any):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.params = list(params)
        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs
        self.optimizer = None

        super().__init__(self.params, self.kwargs)

    def step(self, closure = None, **kwargs):
        self.optimizer.step()

        # every rank need to proceed
        n = math.ceil(len(self.params) / self.world_size)
        for idx, p in enumerate(self.params):
            dist.broadcast(p, src=idx // n)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        super().add_param_group(param_group)

        params = param_group["params"]
        n = math.ceil(len(params) / self.world_size)
        start = self.rank * n
        end = min((self.rank + 1)*n, len(params))
        sharded_param_group = {k: v for k, v in param_group.items()}
        sharded_param_group["params"] = params[start:end]
        if not self.optimizer:
            self.optimizer = self.optimizer_cls(params[start:end], **self.kwargs)
        else:
            self.optimizer.add_param_group(sharded_param_group)
