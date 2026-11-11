from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, softmax
from cs336_basics.optimizer import AdamW
import timeit
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_random_inf(batch_size, context_length):
    in_features = torch.randint(0, 10000, size=(batch_size,context_length))
    return in_features.to(device)

def benchmarking_script(context_length, num_layers, num_heads, d_model, d_ff, warm_up_iter, steps, dtype=None, refer_only=True, rope_theta = 10000.0, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01):

    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=context_length,
        num_layers=num_layers,
        num_heads=num_heads,
        d_model=d_model,
        d_ff=d_ff,
        rope_theta=rope_theta
    )
    model.to(device)
    opt = AdamW(
        params=model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay
    )

    batch_size = 4
    for _ in range(warm_up_iter):
        data = generate_random_inf(batch_size, context_length + 1)
        x = data[:, :-1]
        y = data[:, 1:]
        if(dtype):
            with torch.autocast(device_type='cuda', dtype=dtype):
                logits = model(x)
        else:
            logits = model(x)
        if refer_only == False:
            loss = cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    ff_t = []
    bk_t = []
    op_t = []
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    for i in range(steps):
        data = generate_random_inf(batch_size, context_length + 1)
        x = data[:, :-1]
        y = data[:, 1:]
        t0 = timeit.default_timer()
        if(dtype):
            with torch.autocast(device_type='cuda', dtype=dtype):
                logits = model(x)
        else:
            logits = model(x)
        if refer_only == False:
            loss = cross_entropy(logits, y)
        # if logits.device == 'cuda': torch.cuda.synchronize()
        # t1 = timeit.default_timer()
            opt.zero_grad()
            loss.backward()
            # if logits.device == 'cuda': torch.cuda.synchronize()
            # t2 = timeit.default_timer()
            opt.step()
        # if logits.device == 'cuda': torch.cuda.synchronize()
        # t3 = timeit.default_timer()
        # print(f'step{i+1}:前向传播用时{(t1-t0) * 1000:.4f}ms,反向传播用时{(t2-t1) * 1000:.4f}ms,优化器用时{(t3-t2) * 1000:.4f}ms')
        # ff_t.append((t1-t0) * 1000)
        # bk_t.append((t2-t1) * 1000)
        # op_t.append((t3-t2) * 1000)
    # nff_t, nbk_t, nop_t = torch.tensor(ff_t),torch.tensor(bk_t),torch.tensor(op_t)
    # print(f'前向传播均值:{nff_t.mean(dim=-1):.4f}ms,标准差：{nff_t.std(dim=-1):.4f}ms,反向传播均值:{nbk_t.mean(dim=-1):.4f}ms,标准差：{nbk_t.std(dim=-1):.4f}ms,优化器用时均值:{nop_t.mean(dim=-1):.4f}ms,标准差：{nop_t.std(dim=-1):.4f}ms')
    if refer_only == True:
        torch.cuda.memory._dump_snapshot("memory_snapshot_refer2.pickle")
    else:
        torch.cuda.memory._dump_snapshot("memory_snapshot_full2.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)
benchmarking_script(context_length=128, num_layers=12, num_heads=12, d_model=768, d_ff=3072, warm_up_iter=5, steps=10, refer_only=True)

        