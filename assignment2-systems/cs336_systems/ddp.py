import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import os

class DDPWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.model = module
        self.sync_handles = []
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        for p in self.model.parameters():
            dist.broadcast(p.grad)

        for p in self.model.parameters():
            p.register_post_accumulate_grad_hook(self.make_hook(p))

    def make_hook(self):
        def hook(p):
            if p.grad is not None:
                handle = dist.all_reduce(p.grad, async_op=True)
                self.sync_handles.append((handle, p))
        return hook

    def forward(self, *inputs, **kwargs):
        self.model(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle, p in self.sync_handles:
            handle.wait()
            p.grad._div(self.world_size)

        self.sync_handles.clear() 