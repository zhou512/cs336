import timeit
import torch
import torch._dynamo as dynamo
from cs336_basics.my_model import SingleSelfAttention

# ------------------ 配置 ------------------
batch_size = 8
d_models = [16, 32, 64, 128]
seqs = [256, 1024, 4096]
warmup = 5
loop = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_compile = False
is_cuda = (device == 'cuda')

# recompile_limit 默认是 8；我们在每组结束时 reset dynamo，
# 所以这里保持默认即可。如果仍想留点余量，可以调大：
# dynamo.config.recompile_limit = 16


def sync():
    if is_cuda:
        torch.cuda.synchronize()


for d_model in d_models:
    for seq in seqs:
        print(f'--- d_model={d_model}, seq={seq} ---')
        x = torch.randn(batch_size, seq, d_model, device=device)

        # ============================================================
        # 前向：eval 模式 + no_grad
        # ============================================================
        eval_model = SingleSelfAttention(d_model, context_length=seq, device=device)
        eval_model.eval()
        if use_compile:
            eval_model = torch.compile(eval_model)

        with torch.no_grad():
            for _ in range(warmup):
                eval_model(x)
            sync()
            t0 = timeit.default_timer()
            for _ in range(loop):
                eval_model(x)
                sync()
            t1 = timeit.default_timer()

        fwd_total = (t1 - t0) * 1000
        print(f'  {loop}次前向: {fwd_total:.4f} ms  (单次 {fwd_total / loop:.4f} ms)')

        del eval_model
        if is_cuda:
            torch.cuda.empty_cache()

        # ============================================================
        # 反向：train 模式 + 梯度
        # ============================================================
        train_model = SingleSelfAttention(d_model, context_length=seq, device=device)
        train_model.train()
        if use_compile:
            train_model = torch.compile(train_model)

        def train_step():
            out = train_model(x)
            loss = out.sum()
            loss.backward()
            train_model.zero_grad(set_to_none=True)

        # warmup（同时完成编译）
        for _ in range(warmup):
            train_step()
            sync()

        # 显存统计（此时只有参数 + 输入张量，梯度已通过 set_to_none 释放）
        if is_cuda:
            torch.cuda.reset_peak_memory_stats()
            mem_before_bwd = torch.cuda.memory_allocated()
            print(f'  反向开始前已用显存: {mem_before_bwd / 2**20:.1f} MiB')

        t_bwd_total = 0.0
        for _ in range(loop):
            out = train_model(x)
            loss = out.sum()
            sync()  # 前向结束，开始计时 backward

            tb0 = timeit.default_timer()
            loss.backward()
            sync()
            tb1 = timeit.default_timer()

            t_bwd_total += (tb1 - tb0) * 1000
            train_model.zero_grad(set_to_none=True)

        print(f'  {loop}次反向: {t_bwd_total:.4f} ms  (单次 {t_bwd_total / loop:.4f} ms)')
        if is_cuda:
            peak = torch.cuda.max_memory_allocated()
            print(f'  反向峰值显存: {peak / 2**20:.1f} MiB')

        del train_model, x
        if is_cuda:
            torch.cuda.empty_cache()

        # ============================================================
        # 关键：清掉 dynamo 在 forward code object 上累积的缓存，
        # 避免下一组 (d_model, seq) 的 shape/grad_mode 变化触发
        # recompile_limit（默认 8）被耗尽。
        # ============================================================
        dynamo.reset()

print('脚本运行完毕')