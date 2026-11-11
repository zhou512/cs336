import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit
TENSOR_SIZE = [(1024, 1024), (2048, 2048), (4096, 4096)]
process_num = [2, 4, 6]
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    

def distributed_demo(rank, world_size, tensor_size):
    setup(rank, world_size)
    data = torch.randint(0, 10, tensor_size)
    # print(f"rank {rank} data (before all-reduce): {data}")
    for _ in range(5):
        dist.all_reduce(data, async_op=False)
    dist.barrier()
    t0 = timeit.default_timer()
    dist.all_reduce(data, async_op=False)
    t1 = timeit.default_timer()
    print(f'all-reduce数据大小: {tensor_size[0] * tensor_size[1] * 4 / (2**20):.4f}MB,并行进程数: {world_size}, 用时: {(t1 - t0) * 1000:.4f}ms')
    # print(f"rank {rank} data (after all-reduce): {data}")

if __name__ == "__main__":
    for tensor_size in TENSOR_SIZE:
        for pn in process_num:
            mp.spawn(fn=distributed_demo, args=(pn,tensor_size), nprocs=pn, join=True)