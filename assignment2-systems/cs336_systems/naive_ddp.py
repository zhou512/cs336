import torch
import torch.nn as nn
import torch.distributed as dist
import torch.functional as F
import torch.multiprocessing as mp
import torch.optim
import os
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def naive_ddp(rank, world_size, flat):
    setup(rank, world_size)
    n = 128
    d = world_size
    assert n % d == 0
    local_batch_size = n // d
    model = nn.Linear(10, 10).to(rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    global_inputs = torch.randn((n, 10)).to(rank)
    global_targets = torch.randn((n, 10)).to(rank)

    start_idx = rank * local_batch_size
    end_idx = (rank + 1) * local_batch_size

    local_inputs = global_inputs[start_idx: end_idx]
    local_targets = global_targets[start_idx: end_idx]

    local_logits = model(local_inputs)
    local_loss = nn.MSELoss()(local_logits, local_targets)
    local_loss.backward()
    
    if flat == True:
        grads = [p.grad for p in model.parameters()]
        bucket = _flatten_dense_tensors(grads)
        dist.all_reduce(bucket)
        bucket /= world_size
        for g, s in zip(grads, _unflatten_dense_tensors(bucket, grads)):
            g.data.copy_(s)
    else:
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad)
                p.grad /= world_size

    optimizer.step()
    optimizer.zero_grad()
    print(f"Rank {rank} loss: {local_loss.item():.4f}, weight sample: {model.weight.data[0][0].item():.4f}")
    dist.destroy_process_group()

    if __name__ == "__main__":
        world_size = 4
        mp.spawn(fn=naive_ddp, args=(world_size, ), nprocs=world_size, join=True)

    