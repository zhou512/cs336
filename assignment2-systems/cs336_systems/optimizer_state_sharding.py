import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Type, Any
class OptimizerStateSharding(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        if isinstance(params, torch.Tensor):
            params = [params]
        params = list(params)

        if isinstance(params[0], dict):
            self.global_param_groups = [dict(p) for p in params]
        else:
            self.global_param_groups = [{'params': p} for p in params]

        self.local_param_groups = self.get_shard_groups(self.global_param_groups)
        self.optimizer = optimizer_cls(self.local_param_groups, **kwargs)
        super().__init__(self.optimizer.param_groups, self.optimizer.defaults)


    def get_owner_rank(self, global_idx):
        return global_idx // self.local_param_nums
    
    def get_shard_groups(self, params): #这里传入的params是上面已经初始化好的global_param_groups
        #所有参数的列表，用来分片
        self.global_params = []
        for group in params:
            self.global_params.extend(group['params'])
        param_nums = len(self.global_params)
        self.local_param_nums = (param_nums + self.world_size - 1) // self.world_size
        start_idx = self.rank * self.local_param_nums
        end_idx = min((self.rank + 1) * self.local_param_nums, param_nums)
        self.local_param_ids = [id(p) for p in self.global_params[start_idx:end_idx]]
        #重新构造成参数组的形式
        local_groups = []
        for group in params:
            local_param = [p for p in group['params'] if id(p) in self.local_param_ids]
            if local_param:
                new_group = {k: v for k,v in group.items() if k != 'params'}
                new_group['params'] = local_param
                local_groups.append(new_group)

        return local_groups

    def step(self, closure=None, **kwargs):
        loss = self.optimizer.step(closure=closure, **kwargs)
        i = 0
        for group in self.global_param_groups:
            for p in group['params']:
                owner_rank = self.get_owner_rank(i)
                dist.broadcast(p.data, src=owner_rank)
        """ flat_p = torch.cat([p.data.flatten() for p in self.global_params])
        global_buffer = [torch.empty_like(flat_p) for _ in range(self.world_size)]
        dist.all_gather(global_buffer, flat_p)
        offset = 0
        for i, p in enumerate(self.global_params):
            owner_rank = self.get_owner_rank(i)
            p_num = p.numel()
            p.data.copy_(global_buffer[owner_rank][offset:offset + p_num].view_as(p))
            offset += p_num """

        return loss

    def add_param_group(self, param_group: dict[str, Any]):
        if isinstance(param_group['params'], torch.Tensor):
            param_group['params'] = [param_group['params']]
            
        self.global_param_groups.append(param_group)
        
        # 修正：动态添加参数时，不能直接覆盖已存在的优化器状态。
        # 简单且安全的做法是：针对新添加的参数，再次按 rank 轮询分配（Round-Robin），并只添加新参数到底层优化器
        local_new_params = []
        for p in param_group['params']:
            # 保持轮询分片，避免影响老参数归属
            global_idx = len(self.global_params)
            if global_idx % self.world_size == self.rank:
                local_new_params.append(p)
            self.global_params.append(p)
            
        if local_new_params:
            new_local_group = {k: v for k, v in param_group.items() if k != 'params'}
            new_local_group['params'] = local_new_params
            self.optimizer.add_param_group(new_local_group)
            
        # 同步父类的参数组，确保 lr_scheduler 正常
        self.param_groups = self.optimizer.param_groups
