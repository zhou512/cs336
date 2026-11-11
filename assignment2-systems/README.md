# 作业 2：Systems

profiling、Triton kernel 和分布式训练。自己的实现在 `cs336_systems/` 下。

## 文件

| 文件 | 内容 |
|---|---|
| `flash_attention.py` | FlashAttention-2：Triton 前向 kernel（在线 softmax）+ 两个反向 kernel（dK/dV 一遍、dQ 一遍），另有纯 PyTorch 的分块参考实现，用来对数值 |
| `fatest.py` | Triton 与 PyTorch 两种 attention 的前向 / 反向基准，扫 `seq_len × head_dim × dtype` |
| `pytorch_attention.py` | 朴素 attention 的耗时与峰值显存扫描 |
| `benchmark.py` | Transformer 训练一步的耗时 / 显存基准（不同层数、宽度、上下文长度） |
| `multiple_process.py` | 多进程下 all-reduce 的耗时，按张量大小与进程数扫描 |
| `naive_ddp.py` | 最朴素的 DDP：逐参数 all_reduce 与展平成一个桶的对比 |
| `ddp.py` | hook 版 DDP：梯度算完就发异步 all_reduce，未完成 |
| `fsdp.py` | FSDP：参数 all-gather、梯度写桶后 reduce_scatter，未完成 |
| `optimizer_state_sharding.py` | 优化器状态按 rank 分片，step 后按 owner rank 广播，未完成 |

## 跑法

需要 CUDA GPU 和 triton，依赖与作业 1 相同（`torch`、`einops`、`triton`）。
在 `cs336_systems/` 目录下直接跑，例如：

```bash
python fatest.py
python naive_ddp.py
```

## 状态

FlashAttention-2 的前向和反向都写完了，可以和 PyTorch 实现对比数值、看速度；
DDP / FSDP / 优化器状态分片还在写，hook 时机和分桶边界在调，没全部跑通。
