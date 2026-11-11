import torch
import torch.distributed as dist
import torch.nn as nn

class FSDP(nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype if compute_dtype is not None else torch.float32
        self.target_layers = []
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_weights = {}
        for name, layer in module.named_modules():
            if isinstance(layer, (nn.Linear, nn.Embedding)):
                self.target_layers.append((name, layer))
        for name, layer in self.target_layers:
            w = layer.weight.data
            pad = (-w.shape[0]) % self.world_size
            if pad:
                w = torch.cat([w, w.new_zeros(pad, *w.shape[1:])], dim=0)
            shard = w.chunk(self.world_size)[self.rank].clone().contiguous()
            layer.weight = nn.Parameter(shard)
            self.local_weights[name] = layer.weight
            layer.register_forward_pre_hook()
            layer.register_forward_pre_hook(self.make_pre_hook(name))
            layer.register_forward_hook(self.make_forward_hook(name))
            layer.register_full_backward_pre_hook(self.make_pre_hook(name))
            layer.register_full_backward_hook(self.make_full_backward_hook(name))
        self.buffer_size = 50 * 1024 * 1024
        self.buffer = torch.empty(self.buffer_size, dtype=self.compute_dtype, device='cuda')
        self.buffer_offset = 0
        self.shard_meta = []
        self.sync_handles = []
        self.shard_outputs = []
        self.shard_sizes = []
        self.exec_order = []
        
    def make_record_hook(self, )

    def make_pre_hook(self, name):
        def hook(layer: nn.Module, input):
            weights = self.local_weights[name]
            self.old_dtype = weights.dtype
            out_shape = list(weights.shape)
            out_shape[0] *= self.world_size
            output = torch.empty(out_shape, dtype=weights.dtype, device=weights.device)
            if self.compute_dtype is not None:
                weights = weights.to(self.compute_dtype)
            dist.all_gather_into_tensor(output.reshape(-1), weights.reshape(-1))
            layer.weight.data = output
        return hook

    def make_forward_hook(self, name):
        def hook(layer, input, output):
            layer.weight.data = self.local_weights[name]
        return hook

    def wait_handles(self):
        for handle in self.sync_handles:
            handle.wait()
        self.sync_handles.clear()

    def make_full_backward_hook(self, name):
        def hook(layer, input, output):
            layer.weight.data = self.local_weights[name]
            self.wait_handles()
            #每个分片的梯度已经计算好了，要分片把它放入buffer里面，不同rank对应的buffer不同
            if layer.weight.grad is not None:
                grad = layer.weight.grad
            else:
                return
            flat_grad = grad.view(-1)
            g_num = flat_grad.numel()
            if g_num > self.buffer_size:
                cnt = 0
                self.reduce_scatter()
                while(g_num - cnt > self.buffer_size):
                    self.buffer[:].copy_(flat_grad[cnt:cnt+self.buffer_size])
                    self.reduce_scatter()
                    cnt += self.buffer_size
            if self.buffer_offset + g_num > self.buffer_size:
                self.scatter_reduce()
            #所有layer都会向buffer中写梯度，且写的是全量的梯度，从offset~offset+g_num就是这一个layer对应的梯度的所在位置
            self.buffer[self.buffer_offset: self.buffer_offset + g_num].copy_(flat_grad)
            self.shard_meta.append(
                {
                    'name':name,
                    'g_num':g_num
                }
            )
            self.buffer_offset += g_num
        return hook

    def reduce_scatter(self):
        actual_buffer = self.buffer[:self.buffer_offset]
        self.shard_sizes.append(self.buffer_offset // self.world_size)
        local_grad_shard = torch.empty(self.shard_size, dtype=actual_buffer.dtype, device=actual_buffer.device)
        #对所有rank的actual_buffer进行reduce，然后分片到local_grad_shard上-->这是一个rank所持有的所有层的梯度的集合
        handle = dist.reduce_scatter_tensor(local_grad_shard, actual_buffer, async_op=True)
        self.sync_handles.append(handle)
        self.shard_outputs.append(local_grad_shard)
        self.buffer_offset = 0

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        self.reduce_scatter()
        for handle in self.sync_handles:
            handle.wait()
        self.sync_handles.clear()
        shard_offset = 0
        shard_index = 0
        #字典是顺序存储的，所以自然是顺序遍历
        for meta in self.shard_meta:
            numel = meta['g_num']
            shard_numel = numel // self.world_size
            if(shard_offset + shard_numel > self.shard_sizes[shard_index]):
                shard_index += 1
                shard_offset = 0
            shard_grad = self.shard_outputs[shard_index][shard_offset: shard_offset + shard_numel]
            shard_offset += shard_numel

            name = meta['name']
            local_param = self.local_weights[name]
            local_param.grad = shard_grad.view_as(local_param)
        self.shard_meta.clear()
        self.shard_outputs.clear()
        self.shard_sizes.clear()
        self.buffer_offset = 0
