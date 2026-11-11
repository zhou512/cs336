import torch
import torch.nn.functional as F
import triton
import triton.testing
import math

# 假设你的类代码已经在当前文件或 import 进来了
from flash_attention import FlashAttention_pt, FlashAttention_tr

# ---------------------------------------------------------
# 1. 定义普通的 PyTorch 实现 (用于对比)
# ---------------------------------------------------------
def pytorch_attention(q, k, v, is_causal=False):
    scale = q.shape[-1] ** -0.5
    # (b, nq, d) @ (b, d, nk) -> (b, nq, nk)
    s = torch.matmul(q, k.transpose(-1, -2)) * scale
    if is_causal:
        nq, nk = q.shape[1], k.shape[1]
        mask = torch.tril(torch.ones(nq, nk, device=q.device, dtype=torch.bool))
        s = s.masked_fill(~mask, float('-inf'))
    p = torch.softmax(s, dim=-1)
    o = torch.matmul(p, v)
    return o

# ---------------------------------------------------------
# 2. 定义 Benchmark 工具
# ---------------------------------------------------------
def benchmark_impl(impl_fn, q, k, v, is_causal, is_custom_func=False):
    # 为了算反向传播，需要 requires_grad
    q = q.clone().detach().requires_grad_(True)
    k = k.clone().detach().requires_grad_(True)
    v = v.clone().detach().requires_grad_(True)
    dO = torch.randn_like(q)

    # 包装 forward
    def fwd():
        if is_custom_func:
            return impl_fn.apply(q, k, v, is_causal)
        else:
            return impl_fn(q, k, v, is_causal=is_causal)
    
    # 测前向
    fwd_latency = triton.testing.do_bench(fwd, warmup=10, rep=50)
    
    # 包装 backward (整个 fwd + bwd)
    def fwd_bwd():
        o = fwd()
        o.backward(dO, retain_graph=True)
    
    # 测前向+反向
    total_latency = triton.testing.do_bench(fwd_bwd, warmup=10, rep=50)
    # 纯反向时间 = 总时间 - 前向时间
    bwd_latency = total_latency - fwd_latency

    return fwd_latency, bwd_latency, total_latency

# ---------------------------------------------------------
# 3. 测试主循环 (按题目要求笛卡尔积)
# ---------------------------------------------------------
seq_lens = [128, 512, 2048, 4096, 8192]
embed_dims = [16, 32, 64, 128]
dtypes = [torch.bfloat16, torch.float32]

device = 'cuda' # B200
batch_size = 1

print(f"{'Impl':<10} | {'N':<6} | {'D':<4} | {'Dtype':<15} | {'Fwd(ms)':<10} | {'Bwd(ms)':<10} | {'Total(ms)':<10}")
print("-" * 85)

for dtype in dtypes:
    for n in seq_lens:
        for d in embed_dims:
            q = torch.randn(batch_size, n, d, device=device, dtype=dtype)
            k = torch.randn(batch_size, n, d, device=device, dtype=dtype)
            v = torch.randn(batch_size, n, d, device=device, dtype=dtype)
            
            # 跑 Triton 实现
            try:
                fwd, bwd, total = benchmark_impl(FlashAttention_tr, q, k, v, True, is_custom_func=True)
                print(f"{'Triton':<10} | {n:<6} | {d:<4} | {str(dtype).split('.')[-1]:<15} | {fwd:<10.3f} | {bwd:<10.3f} | {total:<10.3f}")
            except Exception as e:
                print(f"{'Triton':<10} | {n:<6} | {d:<4} | {str(dtype).split('.')[-1]:<15} | Error: {e}")

            # 跑普通 PyTorch 实现 (遇到大序列长可能 OOM，所以加 try-except)
            # 注意：当 N 非常大时，普通 PyTorch 必定 OOM，这里为了不中断程序用 try 包裹
            try:
                # 为了防止跑很久，可以根据需要把 N>8192 的 PyTorch 实现 skip 掉
                if n <= 16384: 
                    fwd, bwd, total = benchmark_impl(pytorch_attention, q, k, v, True, is_custom_func=False)
                    print(f"{'PyTorch':<10} | {n:<6} | {d:<4} | {str(dtype).split('.')[-1]:<15} | {fwd:<10.3f} | {bwd:<10.3f} | {total:<10.3f}")
                else:
                    print(f"{'PyTorch':<10} | {n:<6} | {d:<4} | {str(dtype).split('.')[-1]:<15} | OOM (Skipped)")
            except torch.cuda.OutOfMemoryError:
                print(f"{'PyTorch':<10} | {n:<6} | {d:<4} | {str(dtype).split('.')[-1]:<15} | OOM (Skipped)")
                torch.cuda.empty_cache()
            
            # 清空显存
            del q, k, v
            torch.cuda.empty_cache()