
# Lecture 3 笔记：LLM 架构与超参数

> 主题：现代 Transformer 到底长什么样？哪些设计是共识？哪些是变体？为什么大家这样选？

---

## 0. 一句话总结

现代 LLM 基本是 **LLaMA-like 架构**：

- **Pre-Norm / RMSNorm**
- **SwiGLU / GeGLU**
- **RoPE 位置编码**
- **线性层和 Norm 通常无 bias**
- **Serial block 为主**
- 超参数总体保守：`d_ff ≈ 4 d_model`，GLU 变体 `≈ 8/3 d_model`，`head_dim * num_heads ≈ d_model`，aspect ratio 约 100–200
- 稳定性技巧：z-loss、QK norm、logit soft-capping、梯度裁剪等

---

## 1. 原始 Transformer vs 现代简单 Transformer

| 组件 | 原始 Transformer | 现代常见做法 |
|---|---|---|
| 位置编码 | 正弦/余弦 | RoPE |
| FFN 激活 | ReLU | SwiGLU / GeGLU |
| Norm | Post-Norm LayerNorm | Pre-Norm RMSNorm |
| Bias | 有 bias | 通常无 bias |
| Block | 串行 | 多数串行，少数并行 |
| 输出 | Linear + Softmax | 类似，但稳定性技巧更多 |

现代 block 常见结构：

```text
x -> Norm -> Causal Multi-Head Self-Attention w/ RoPE -> Add
  -> Norm -> Position-wise Feed-Forward -> Add
```

核心变化原因：**更稳定、更快、更容易 scale**。

---

## 2. Pre-Norm vs Post-Norm

### 原始：Post-Norm

- LayerNorm 放在残差相加之后。
- 缺点：梯度衰减、训练不稳定、常需 warmup。
- 深层时梯度容易在浅层变小。

### 现代：Pre-Norm

- LayerNorm 放在子层输入前，残差连接直通。
- 优点：梯度传播更好、loss spike 更少、可用更大学习率。
- 结论：**几乎所有人都用非残差 Norm，通常是 Pre-Norm。**

### 新变体：Non-residual Post-Norm / Double Norm

- 不在残差流里做 Norm，而是在残差流外做 Post-Norm。
- 代表：Grok、Gemma 2、Olmo 2 等。
- 目的：保留残差直通，同时改善稳定性。

---

## 3. LayerNorm vs RMSNorm

### LayerNorm

\[
y = \frac{x - \mathbb{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} \gamma + \beta
\]

- 减均值、除方差、有 bias。
- 原始 Transformer、GPT-1/2/3、OPT、BLOOM 等使用。

### RMSNorm

\[
y = \frac{x}{\sqrt{||x||_2^2 + \epsilon}} \gamma
\]

- 不减均值、无 bias。
- 代表：LLaMA、PaLM、Mistral、Qwen、DeepSeek 等现代模型。

### 为什么 RMSNorm？

- 操作更少，参数更少。
- 但关键不是 FLOPs，而是 **runtime 和数据移动**。
- 矩阵乘占 99.8% FLOPs，但 Norm 可能占约 25% runtime。
- RMSNorm 减少内存搬运，实际 wall-clock 更快。

### 顺带：去掉 bias

现代模型常去掉线性层和 Norm 的 bias：

- 省内存/参数。
- 对优化稳定性也有好处。

---

## 4. 激活函数与 FFN

### 常见激活

- ReLU：原始 Transformer、T5、Gopher、Chinchilla、OPT
- GeLU：GPT-1/2/3、GPT-J、GPT-NeoX、BLOOM
- SwiGLU / GeGLU：Llama、PaLM、T5 v1.1、多数 2023 后模型

### GLU 变体

普通 FFN：

\[
\mathrm{FFN}(x) = \max(0, xW_1)W_2
\]

GLU 变体加入门控：

\[
\mathrm{FFN}_{\mathrm{ReGLU}}(x) = (\max(0, xW_1) \otimes xV)W_2
\]

GeGLU：

\[
\mathrm{FFN}_{\mathrm{GeGLU}}(x) = (\mathrm{GELU}(xW) \otimes xV)W_2
\]

SwiGLU：

\[
\mathrm{FFN}_{\mathrm{SwiGLU}}(x) = (\mathrm{Swish}(xW) \otimes xV)W_2
\]

### 结论

- GLU 变体通常一致更好。
- **SwiGLU / GeGLU 是当前主流。**
- 注意：GLU 会让 FFN 中间维度通常变成 `8/3 d_model`，而不是 `4 d_model`。

---

## 5. Serial vs Parallel Layers

标准 Serial：

```text
x -> Attention -> MLP
```

Parallel：

\[
y = x + \mathrm{MLP}(\mathrm{LayerNorm}(x)) + \mathrm{Attention}(\mathrm{LayerNorm}(x))
\]

- 代表：GPT-J、PaLM、GPT-NeoX。
- 优点：可融合矩阵乘，训练约快 15%。
- 大尺度下质量基本中性。
- 但当前多数模型仍用 **Serial**。

---

## 6. 位置编码

### 类型

| 类型 | 做法 | 代表 |
|---|---|---|
| 正弦 | 加 sin/cos | 原始 Transformer |
| 绝对 | 学习位置向量加到 embedding | GPT-1/2/3、OPT |
| 相对 | 在 attention 中加入相对位置 | T5、Gopher、Chinchilla |
| RoPE | 旋转 query/key | LLaMA、PaLM、多数 2024+ 模型 |

### RoPE 核心思想

希望 attention 只依赖相对位置：

\[
\langle f(x,i), f(y,j) \rangle = g(x,y,i-j)
\]

做法：把 q/k 的坐标成对旋转，旋转角度与位置有关。

- 不是加法，而是乘法/旋转。
- 没有交叉项。
- 在每个 attention 层对 q/k 应用。
- 现代主流：RoPE。

### 变体

- 部分旋转：如 Gemma 4 只旋转部分维度。
- 交错注意力：Full attention + Sliding Window Attention，长程用 NoPE/Full，短程用 RoPE/SWA。

---

## 7. 超参数：哪些重要，哪些不重要？

### 7.1 FFN 维度比：`d_ff / d_model`

| 模型 | 比例 |
|---|---|
| 默认 | 4 |
| GLU 变体 | 8/3 ≈ 2.67 |
| PaLM | 4 |
| Mistral 7B | 3.5 |
| LLaMA-2 70B | 3.5 |
| LLaMA 70B | 2.68 |
| Qwen 14B | 2.67 |
| DeepSeek 67B | 2.68 |
| Yi 34B | 2.85 |
| T5 v1.1 | 2.5 |

- 经验盆地：1–10 之间都接近最优。
- 默认 `4x` 或 `8/3x` 通常很好。
- 例外：T5 11B 用过 `d_ff = 65536, d_model = 1024`，即 64x，为了 TPU 大矩阵乘效率；后续 T5 v1.1 回到 2.5x。

### 7.2 Head 维度 / 头数 / 模型维度

- 常见：`head_dim * num_heads ≈ d_model`
- 多数模型 ratio 约 1。
- 例外：部分 Google 模型。
- 多头注意力计算效率高，reshape 成 heads 后计算量基本不变。

### 7.3 Aspect Ratio：`d_model / n_layer`

- 常见较好范围：100–200。
- 宽范围都能达到相似性能。
- 深度难并行、延迟高；宽度易并行。
- 实际选择常由系统/并行约束决定。

### 7.4 词表大小

| 类型 | 大小 | 例子 |
|---|---|---|
| 单语 | 30–50k | 原始 37k、GPT-2/3 50257、T5 32128、LLaMA 32000 |
| 多语/生产 | 100–250k | mT5 250k、PaLM 256k、GPT-4 100276、Gemma 4 262144、DeepSeek 100k、Qwen 152064、Yi 64000 |

### 7.5 Dropout 与 Weight Decay

- 旧模型常用 dropout 0.1。
- 新模型多数不用 dropout，只保留 weight decay 0.1。
- Weight decay 主要不是防过拟合，而是影响优化动态，和 LR schedule 交互。

---

## 8. 稳定性技巧

### 8.1 z-loss

让 softmax 的归一化项 `log Z` 接近 0：

\[
L = \sum_i \left[\log P(x_i) - \alpha \log^2 Z(x_i)\right]
\]

PaLM 使用类似技巧：

\[
z\_loss = 10^{-4} \cdot \log^2 Z
\]

代表：PaLM、Baichuan 2、DCLM、OLMo 2/3。

### 8.2 QK Norm

- 对 query/key 做 LayerNorm。
- 防止 attention softmax logits 爆炸。
- 代表：DCLM、OLMo2、Gemma 2、Qwen3、Gemma 4。

### 8.3 Logit Soft-Capping

\[
\mathrm{logits} \leftarrow \mathrm{soft\_cap} \cdot \tanh(\mathrm{logits} / \mathrm{soft\_cap})
\]

- attention 层 `soft_cap = 50.0`
- final 层 `soft_cap = 30.0`
- 防止 logits 爆炸，但可能有性能影响。

### 8.4 其他

- 梯度裁剪
- embedding weight decay
- 避免 loss spike：不要训练出 OLMo 那种蓝色剧烈震荡曲线

---

## 9. 注意力头变体

### MHA -> MQA -> GQA -> MLA

| 方法 | 思路 | 特点 |
|---|---|---|
| MHA | 每个 head 独立 Q/K/V | 标准，KV cache 大 |
| MQA | 多 Q，1 组 K/V | KV cache 小，可能有小 PPL 损失 |
| GQA | 分组 K/V | 性能接近 MHA，推理更快 |
| MLA | 多头潜在注意力 | DeepSeek v2 等 |

### 稀疏 / Sliding Window Attention

- 全注意力对长上下文太贵。
- 用稀疏/滑窗限制 attention 模式。
- 常见：交错 Full + Sliding Window。
  - 如 Command A：每 4 层一个 full attention。
  - LLaMA 4、Gemma 3/4、OLMo 3 等也有类似设计。
- 长程信息用 Full/NoPE，短程信息用 RoPE + SWA。

---

## 10. 最终共识清单

### 架构

- Pre-Norm / RMSNorm
- 无 bias
- SwiGLU / GeGLU
- RoPE
- Serial block
- 多数模型 LLaMA-like

### 超参数

- `d_ff = 4 d_model`，GLU 为 `8/3 d_model`
- `head_dim * num_heads ≈ d_model`
- Aspect ratio 100–200
- 单语词表 30–50k，多语 100–250k
- 新模型常无 dropout，weight decay 0.1

### 稳定性

- z-loss
- QK norm
- logit soft-capping
- 梯度裁剪
- 注意 softmax 数值稳定性

### 主要差异

- 位置编码
- 激活函数
- tokenization
- attention 模式：GQA/MQA、稀疏/滑窗、混合注意力

---

## 11. 如果只记 10 条

1. 现代 LLM ≈ LLaMA-like。
2. Pre-Norm 比 Post-Norm 稳。
3. RMSNorm 主流，因为省数据移动。
4. 去掉 bias 是常态。
5. SwiGLU/GeGLU 是主流 FFN。
6. RoPE 是主流位置编码。
7. `d_ff ≈ 4 d_model`，GLU 用 `8/3`。
8. `head_dim * num_heads ≈ d_model`。
9. 稳定性靠 z-loss、QK norm、soft-capping。
10. GQA/MQA 是为了省推理时 KV cache。