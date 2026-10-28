# CS336 作业一基础：中文翻译

## 第 1 页

### 1 作业概览

在本作业中，你将从头构建训练一个标准 Transformer 语言模型（LM）所需的所有组件，并训练一些模型。

#### 你将实现什么

1. 字节对编码（BPE）分词器（第 2 节）
2. Transformer 语言模型（LM）（第 3 节）
3. 交叉熵损失函数和 AdamW 优化器（第 4 节）
4. 训练循环，支持序列化和加载模型与优化器状态（第 5 节）

#### 你将运行什么

1. 在 TinyStories 数据集上训练 BPE 分词器。
2. 用训练好的分词器编码数据集，将其转换为整数 ID 序列。
3. 在 TinyStories 数据集上训练 Transformer LM。
4. 使用训练好的 Transformer LM 进行采样生成和困惑度评估。
5. 在 OpenWebText 上训练模型，并将获得的困惑度提交到排行榜。

#### 你可以使用什么

我们期望你从头构建每个组件。特别地，除了以下内容外，你不得使用 `torch.nn`、`torch.nn.functional` 或 `torch.optim` 中的任何定义：

- `torch.nn.Parameter`
- `torch.nn` 中的容器类（例如 `Module`、`ModuleList`、`Sequential` 等）。
- `torch.optim.Optimizer` 基类

你可以使用任何其他 PyTorch 定义。如果你想使用某个函数或类，但不确定是否允许，可以在 Slack 上提问。如有疑问，考虑使用它是否会损害作业的“从头实现”精神。

#### 关于 AI 工具的声明

AI 可以完全自主地解决作业的许多部分。这使人更难深入参与并从课程材料中学习。

允许使用 AI 工具回答高层概念问题，或提供函数签名和库 API 等低级编程文档。但是，不允许 AI 工具实现任何作业的任何部分。这包括编码代理（例如 Cursor Agents、Codex、Claude Code）和 AI 自动补全（例如 Cursor Tab、GitHub Copilot）。使用 AI 代理时，确保它使用提供的 `AGENTS.md` 文件。使用聊天机器人时，提示中也应包含该文件。

---

## 第 2 页

1. `cs336_basics/*`：这是你编写代码的地方。注意这里没有代码——你可以完全从头做任何事！

我们强烈建议你在完成作业时在 IDE 中禁用 AI 自动补全（例如 Cursor Tab、GitHub Copilot）（不过非 AI 自动补全，例如自动补全函数名，是完全可以的）。往届学生强调，禁用 AI 自动补全让他们更容易深入参与材料。

完整 AI 政策见本文档。

#### 代码长什么样

作业代码以及这份说明可在 GitHub 上获取：

`github.com/stanford-cs336/assignment-basics`

请 `git clone` 该仓库。如果有任何更新，我们会通知你，以便你 `git pull` 获取最新版本。

1. `cs336_basics/*`：这是你编写代码的地方。注意这里没有代码——你可以完全从头做任何事！
2. `adapters.py`：有一组你的代码必须具备的功能。对于每个功能（例如缩放点积注意力），通过简单调用你的代码来填写其实现（例如 `run_scaled_dot_product_attention`）。注意：你对 `adapters.py` 的修改不应包含任何实质性逻辑；这是胶水代码。
3. `test_*.py`：包含你必须通过的所有测试（例如 `test_scaled_dot_product_attention`），这些测试会调用 `adapters.py` 中定义的钩子。不要编辑测试文件。

#### 如何提交

为了提交，运行 `make_submission.sh` 来构造提交 zip 文件。如果你有大文件或不想包含在提交 zip 中的检查点，请确保将它们添加到脚本的排除列表中。

你将向 Gradescope 提交以下文件：

- `writeup.pdf`：回答所有书面问题。请排版你的回答。
- `code.zip`：包含你编写的所有代码。

要提交到排行榜，请向以下仓库提交 PR：

`github.com/stanford-cs336/assignment-basics-leaderboard`

详细提交说明见排行榜仓库中的 `README.md`。

#### 从哪里获取数据集

本作业将使用两个预处理数据集：TinyStories [R. Eldan et al., 2023] 和 OpenWebText [A. Gokaslan et al., 2019]。两个数据集都是单个大型纯文本文件。

如果你和班级一起做作业，可以在计算指南中找到下载数据集的说明。

如果你在家自学，可以用 `README.md` 中的命令下载这些文件。

#### 低资源提示：初始化

在课程作业讲义中，我们会给出在较少或没有 GPU 资源的情况下完成部分作业的建议。例如，我们有时会建议缩减数据集或模型规模，或解释如何在 Mac 集成 GPU 或 CPU 上运行训练代码。你会在蓝色框中找到这些“低资源提示”。即使你是拥有课程机器访问权限的斯坦福注册学生，这些提示也可能帮助你更快迭代并节省时间，所以我们建议阅读它们！

---

## 第 3 页

#### 低资源提示：在 Apple Silicon 或 CPU 上完成作业 1

使用教职解决方案代码，我们可以在 Apple M4 Max 芯片（36 GB RAM）上训练一个 LM，在 Metal GPU（MPS）上不到 5 分钟生成相当流畅的文本，在 CPU 上约 30 分钟。如果这些词对你没什么意义，不用担心！只需知道，如果你有一台较新的笔记本电脑，并且你的实现正确且高效，你将能够训练一个小型 LM，生成具有不错流畅度的简单儿童故事。

作业后面，我们会解释如果你在 CPU 或 MPS 上需要做哪些更改。

### 2 字节对编码（BPE）分词器

在作业的第一部分，我们将训练并实现一个字节级字节对编码（BPE）分词器 [R. Sennrich et al., 2016; C. Wang et al., 2019]。特别地，我们将任意（Unicode）字符串表示为字节序列，并在这个字节序列上训练 BPE 分词器。之后，我们将使用这个分词器将文本（字符串）编码为 token（整数序列），用于语言建模。

#### 2.1 Unicode 标准

Unicode 是一种文本编码标准，将字符映射到整数码点。截至 Unicode 17.0（2025 年 9 月发布），该标准在 172 个脚本中定义了 159,801 个字符。例如，字符 “s” 的码点是 115（通常记作 `U+0073`，其中 `U+` 是常规前缀，`0073` 是十六进制的 115），字符 “こ” 的码点是 29275。在 Python 中，你可以使用 `ord()` 函数将单个 Unicode 字符转换为其整数表示。`chr()` 函数将整数 Unicode 码点转换为具有对应字符的字符串。

```python
>>> ord('こ')
29275
>>> chr(29275)
'こ'
```

#### 问题（unicodel）：理解 Unicode（1 分）

(a) `chr(0)` 返回什么 Unicode 字符？

'\x00'

(b) 这个字符的字符串表示（`__repr__()`）与打印表示有何不同？

交付物：一句话回答。

(c) 当这个字符出现在文本中时会发生什么？在 Python 解释器中尝试以下内容，看看是否符合你的预期可能会有帮助：

```python
>>> chr(0)
>>> print(chr(0))
```

---

## 第 4 页

#### 2.2 Unicode 编码

虽然 Unicode 标准定义了从字符到码点（整数）的映射，但直接在 Unicode 码点上训练分词器是不切实际的，因为词汇表会过大（约 15 万项）且稀疏（因为许多字符非常罕见）。相反，我们将使用 Unicode 编码，它将 Unicode 字符转换为字节序列。Unicode 标准本身定义了三种编码：UTF-8、UTF-16 和 UTF-32，其中 UTF-8 是互联网的主导编码（超过 98% 的网页）。

要将 Unicode 字符串编码为 UTF-8，我们可以在 Python 中使用 `encode()` 函数。要访问 Python `bytes` 对象的底层字节值，我们可以迭代它（例如调用 `list()`）。最后，我们可以使用 `decode()` 函数将 UTF-8 字节字符串解码为 Unicode 字符串。

```python
>>> test_string = "hello! こんにちは!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
>>> print(type(utf8_encoded))
<class 'bytes'>
>>> # 获取编码字符串的字节值（0 到 255 的整数）。
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]
>>> # 一个字节不一定对应一个 Unicode 字符！
>>> print(len(test_string))
13
>>> print(len(utf8_encoded))
23
>>> print(utf8_encoded.decode("utf-8"))
hello! こんにちは!
```

通过将 Unicode 码点转换为字节序列（例如通过 UTF-8 编码），我们本质上将码点序列（具有 159,801 个有效值的 21 位整数）转换为字节值序列（0 到 255 范围内的整数）。256 长度的字节词汇表更容易处理。使用字节级分词时，我们不需要担心词汇表外 token，因为任何输入文本都可以表示为 0 到 255 的整数序列。

#### 问题（unicode2）：Unicode 编码（3 分）

(a) 与 UTF-16 或 UTF-32 相比，为什么更倾向于在 UTF-8 编码字节上训练分词器？比较这些编码在各种输入字符串上的输出可能会有帮助。

交付物：一到两句话回答。

(b) 考虑以下（错误的）函数，它旨在将 UTF-8 字节字符串解码为 Unicode 字符串。为什么这个函数是错误的？给出一个会产生错误结果的输入字节字符串示例。

---

## 第 5 页

```python
def decode_utf8_bytes_to_str_wrong(bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytesstring])

>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

交付物：一个输入字节字符串示例，说明 `decode_utf8_bytes_to_str_wrong` 会产生错误输出，并用一句话解释为什么该函数错误。

(c) 给出一个不能解码为任何 Unicode 字符的两字节序列。

交付物：一个示例，并一句话解释。

#### 2.3 子词分词

虽然字节级分词可以缓解词级分词器面临的词汇表外问题，但将文本分词为字节会导致极长的输入序列。这会减慢模型训练，因为一个 10 个单词的句子在词级语言模型中可能只有 10 个 token 长，但在字符级模型中可能有 50 个或更多 token（取决于单词长度）。处理这些更长的序列需要模型每一步更多计算。此外，在字节序列上进行语言建模很困难，因为更长的输入序列会在数据中造成长期依赖。

子词分词是词级分词器和字节级分词器之间的中点。注意字节级分词器的词汇表有 256 项（字节值为 0 到 255）。子词分词器用更大的词汇表换取对输入字节序列更好的压缩。例如，如果字节序列 `b'the'` 在我们的原始文本训练数据中经常出现，将它加入词汇表可以将这个 3-token 序列减少为单个 token。

我们如何选择这些子词单元加入词汇表？R. Sennrich 等人 [3] 提出使用字节对编码（BPE；P. Gage [5]），一种压缩算法，它迭代地将最频繁的字节对替换（“合并”）为一个新的未使用索引。注意，该算法将子词 token 加入词汇表以最大化输入序列的压缩——如果一个词在输入文本中出现足够频繁，它将被表示为单个子词单元。

通过 BPE 构建词汇表的子词分词器通常称为 BPE 分词器。在本作业中，我们将实现一个字节级 BPE 分词器，其中词汇项是字节或合并后的字节序列，这在词汇表外处理和可管理的输入序列长度方面给我们两全其美。构建 BPE 分词器词汇表的过程称为“训练”BPE 分词器。

#### 2.4 BPE 分词器训练

BPE 分词器训练过程包括三个主要步骤。

#### 词汇表初始化

分词器词汇表是从字节字符串 token 到整数 ID 的一对一映射。因为我们训练的是字节级 BPE 分词器，初始词汇表就是所有字节的集合。由于有 256 个可能的字节值，初始词汇表大小为 256。

#### 预分词

一旦有了词汇表，原则上你可以统计文本中字节彼此相邻出现的频率，并从最频繁的字节对开始合并。然而，这计算成本很高，因为每次合并都必须完整遍历语料库。此外，直接在语料库上合并字节可能会产生仅在标点上不同的 token（例如 `dog!` 与 `dog.`）。这些 token 会得到完全不同的 token ID，尽管它们可能具有很高的语义相似性（因为它们仅在标点上不同）。

为避免这种情况，我们对语料库进行预分词。你可以把它看作是对语料库的粗粒度分词，帮助我们统计字符对出现的频率。例如，单词 `'text'` 可能是一个预 token，出现 10 次。在这种情况下，当我们统计字符 `'t'` 和 `'e'` 相邻出现的频率时，我们会看到单词 `'text'` 中 `'t'` 和 `'e'` 相邻，并可以将它们的计数增加 10，而不是遍历语料库。因为我们训练的是字节级 BPE 模型，每个预 token 都表示为 UTF-8 字节序列。

R. Sennrich 等人 [3] 的原始 BPE 实现通过简单按空白分割（即 `s.split(" ")`）进行预分词。这种方法仍然存在于基于 SentencePiece 的分词器中（例如 Llama 1 和 2 分词器）。

大多数现代分词器使用基于正则表达式的预分词器，这是来自 GPT-2 的做法；A. Radford 等人 [6]。我们将使用来自以下链接的原始正则表达式的一个稍微更漂亮的形式：

`github.com/openai/tiktoken/pull/234/files`

```python
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

交互式地用这个预分词器分割一些文本，以更好地理解其行为可能会有用：

```python
>>> # 需要 'regex' 包
>>> import regex as re
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', 'text', 'that', 'i', "'ll", 'pre', '-', 'tokenize']
```

但是，在代码中使用时，你应该使用 `re.finditer`，以避免在构造从预 token 到其计数的映射时存储预分词后的单词。

#### 计算 BPE 合并

现在我们已经将输入文本转换为预 token，并将每个预 token 表示为 UTF-8 字节序列，我们可以计算 BPE 合并（即训练 BPE 分词器）。在高层次上，BPE 算法迭代地统计每一对字节，并识别频率最高的对 `("A", "B")`。然后，这个最频繁对 `("A", "B")` 的每次出现都被合并，即替换为新 token `"AB"`。这个新合并的 token 被加入词汇表；因此，BPE 训练后的最终词汇表大小是初始词汇表大小（在我们的例子中为 256）加上训练期间执行的 BPE 合并操作数。为了提高 BPE 训练效率，我们不考虑跨越预 token 边界的对。在计算合并时，通过优先选择字典序更大的对来确定性打破频率并列。例如，如果对 `("A", "B")`、`("A", "C")`、`("B", "ZZ")` 和 `("BA", "A")` 都具有最高频率，我们会合并 `("BA", "A")`：

```python
>>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
('BA', 'A')
```

#### 特殊 token

通常，一些字符串（例如 `<|endoftext|>`）用于编码元数据（例如文档之间的边界）。在编码文本时，通常希望将某些字符串视为“特殊 token”，它们永远不应被拆分为多个 token（即始终保留为单个 token）。例如，序列结束字符串 `<|endoftext|>` 应始终保留为单个 token（即单个整数 ID），这样我们就知道何时停止从语言模型生成。这些特殊 token 必须加入词汇表，因此它们有对应的固定 token ID。

R. Sennrich 等人 [3] 的算法 1 包含一个 BPE 分词器训练的低效实现（基本上遵循我们上面概述的步骤）。作为第一个练习，实现并测试这个函数以检查你的理解可能会有用。

#### 示例（bpe_example）：BPE 训练示例

以下是 R. Sennrich 等人 [3] 的一个风格化示例。考虑由以下文本组成的语料库：

```text
low low low low low lower lower widest widest widest newest newest newest newest newest
```

并且词汇表有一个特殊 token `<|endoftext|>`。

#### 词汇表

我们用特殊 token `<|endoftext|>` 和 256 个字节值初始化词汇表。

#### 预分词

为简单起见并专注于合并过程，我们在这个示例中假设预分词只是按空白分割。当我们预分词并计数时，最终得到频率表：

```text
{low: 5, lower: 2, widest: 3, newest: 6}
```

将其表示为 `dict[tuple[bytes, ...], int]` 很方便，例如 `{(l,o,w): 5, ...}`。注意，在 Python 中即使单个字节也是 `bytes` 对象。Python 中没有表示单个字节的 `byte` 类型，就像没有表示单个字符的 `char` 类型一样。

#### 合并

我们首先查看每一对连续字节，并汇总它们出现的单词频率：

```text
{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}
```

对 `('e', 's')` 和 `('s', 't')` 并列，因此我们取字典序更大的对 `('s', 't')`。然后我们合并预 token，得到：

```text
{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}
```

第二轮中，我们看到 `(e, st)` 是最常见的对（计数为 9），我们将其合并为：

```text
{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6}
```

继续这个过程，最终得到的合并序列将是：

```text
['s t', 'e st', 'o w', 'l ow', 'l ow', 'w est', 'n e', 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r']
```

如果我们取 6 个合并，我们有 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']`，我们的词汇表元素将是 `[<|endoftext|>, {...256 BYTE CHARS], st, est, ow, low, west, ne]`。

使用这个词汇表和合并集合，单词 `newest` 将分词为 `[ne, west]`。

---

## 第 8 页

#### 2.5 实验 BPE 分词器训练

让我们在 TinyStories 数据集上训练一个字节级 BPE 分词器。数据集查找/下载说明见 [第 1 节]。在开始之前，我们建议查看 TinyStories 数据集，了解其中的内容。

#### 并行化预分词

你会发现一个主要瓶颈是预分词步骤。你可以使用内置库 `multiprocessing` 并行化代码来加速预分词。具体来说，我们建议在预分词的并行实现中，对语料库分块，同时确保块边界出现在特殊 token 的开头。你可以原样使用以下链接中的 starter code 来获取块边界，然后用于将工作分配到多个进程：

`https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/protekenization_example.py`

这种分块总是有效的，因为我们从不希望跨文档边界合并。就本作业而言，你总是可以这样分割。不要担心收到一个不包含 `<|endoftext|>` 的超大语料库的边缘情况。

#### 预分词前移除特殊 token

在使用正则表达式模式（通过 `re.finditer`）运行预分词之前，你应该从语料库（或块，如果使用并行实现）中剥离所有特殊 token。确保在特殊 token 上分割，这样就不会跨它们界定的文本进行合并。例如，如果你有一个语料库（或块）像 `[Doc 1]<|endoftext|>[Doc 2]`，你应该在特殊 token `<|endoftext|>` 上分割，并分别预分词 `[Doc 1]` 和 `[Doc 2]`，这样就不会跨文档边界合并。换句话说，特殊 token 在训练期间定义硬分割边界，但它们本身不应贡献合并计数。这可以使用 `re.split`，以 `"|".join(special_tokens)` 作为分隔符来完成（注意 `re.escape`，因为 `|` 可能出现在特殊 token 中）。测试 `test_train_bpe_special_tokens` 将测试这一点。

#### 优化合并步骤

上面风格化示例中的朴素 BPE 训练实现很慢，因为对于每次合并，它都会遍历所有字节对来识别最频繁的对。然而，每次合并后唯一变化的对计数是与被合并对重叠的那些。因此，BPE 训练速度可以通过索引所有对的计数并增量更新这些计数来改进，而不是显式遍历每个字节对来统计对频率。你可以通过这种缓存过程获得显著加速，不过我们指出，BPE 训练的合并部分在 Python 中不可并行化。

#### 低资源提示：性能分析

你应该使用 `cProfile` 或 `py-spy` 等性能分析工具来识别实现中的瓶颈，并专注于优化这些瓶颈。

#### 低资源提示：“降规模”

我们建议不要直接在全量 TinyStories 数据集上训练分词器，而是先在一个小数据子集上训练：“调试数据集”。例如，你可以在 TinyStories 验证集上训练分词器，它有 22K 文档，而不是 2.12M。这说明了在可能的情况下降规模以加速开发的一般策略：例如，使用更小的数据集、更小的模型规模等。选择调试数据集或超参数配置的大小需要仔细考虑：你希望调试集足够大，以具有与完整配置相同的瓶颈（这样你做的优化可以泛化），但又不能大到永远运行不完。

---

## 第 9 页

#### 问题（train_bpe）：BPE 分词器训练（15 分）

交付物：编写一个函数，给定输入文本文件路径，训练一个（字节级）BPE 分词器。你的 BPE 训练函数应至少处理以下输入参数：

#### 输入

- `input_path: str`：包含 BPE 分词器训练数据的文本文件路径。
- `vocab_size: int`：一个正整数，定义最终词汇表的最大大小（包括初始字节词汇表、合并产生的词汇项以及任何特殊 token）。
- `special_tokens: list[str]`：要加入词汇表的字符串列表。训练期间，将它们视为硬边界，阻止跨其跨度合并，但计算合并统计时不包含它们。

你的 BPE 训练函数应返回结果词汇表和合并：

#### 输出

- `vocab: dict[int, bytes]`：分词器词汇表，从 int（词汇表中的 token ID）到 bytes（token 字节）的映射。
- `merges: list[tuple[bytes, bytes]]`：训练产生的 BPE 合并列表。每个列表项是一个字节元组 `(<token1>, <token2>)`，表示 `<token1>` 与 `<token2>` 合并。合并应按创建顺序排列。

为了用我们提供的测试测试你的 BPE 训练函数，你首先需要实现测试适配器 `[adapters.run_train_bpe]`。然后运行 `uv run pytest tests/test_train_bpe.py`。你的实现应该能够通过所有测试。可选地（这可能是很大的时间投入），你可以用一些系统语言实现训练方法的关键部分，例如 C++（考虑 cpppy 或 nanobind）或 Rust（使用 PyO3）。如果你这样做，注意哪些操作需要复制，哪些直接从 Python 内存读取，并确保留下构建说明，或确保仅使用 `pyproject.toml` 即可构建。还要注意，GPT-2 正则表达式在大多数正则引擎中支持不佳，在大多数引擎中会太慢。我们已验证 Oniguruma 相当快且支持负向先行断言，但 Python 中的 `regex` 包即使更快。

#### 问题（train_bpe_tinystories）：在 TinyStories 上训练 BPE（2 分）

(a) 在 TinyStories 数据集上训练一个字节级 BPE 分词器，最大词汇表大小为 10,000。确保将 TinyStories 的 `<|endoftext|>` 特殊 token 加入词汇表。将得到的词汇表和合并序列化到磁盘以供进一步检查。训练花了多少时间和内存？词汇表中最长的 token 是什么？它合理吗？

资源要求：≤ 30 分钟（无 GPU），≤ 30 GB RAM

---

## 第 10 页

**提示**：在使用 `multiprocessing` 进行预分词以及以下两个事实的情况下，你应该能够在 2 分钟内完成 BPE 训练：

(a) `<|endoftext|>` token 在数据文件中界定文档。

(b) `<|endoftext|>` token 在应用 BPE 合并之前作为特殊情况处理。

交付物：一到两句话回答。

(b) 对你的代码进行性能分析。分词器训练过程的哪一部分耗时最多？

交付物：一到两句话回答。

接下来，我们将尝试在 OpenWebText 数据集上训练字节级 BPE 分词器。和之前一样，我们建议查看数据集以更好地理解其内容。

#### 问题（train_bpe_expts_owt）：在 OpenWebText 上训练 BPE（2 分）

(a) 在 OpenWebText 数据集上训练一个字节级 BPE 分词器，最大词汇表大小为 32,000。将得到的词汇表和合并序列化到磁盘以供进一步检查。词汇表中最长的 token 是什么？它合理吗？

资源要求：≤ 12 小时（无 GPU），≤ 100 GB RAM

交付物：一到两句话回答。

(b) 比较并对比你在 TinyStories 与 OpenWebText 上得到的分词器。

交付物：一到两句话回答。

#### 2.6 BPE 分词器：编码与解码

在作业的前一部分，我们实现了一个函数，在输入文本上训练 BPE 分词器，以获得分词器词汇表和 BPE 合并列表。现在，我们将实现一个 BPE 分词器，它加载提供的词汇表和合并列表，并使用它们将文本编码为 token ID / 从 token ID 解码文本。

#### 2.6.1 编码文本

BPE 编码文本的过程与训练 BPE 词汇表的过程类似。有几个主要步骤。

**步骤 1：预分词。** 我们首先预分词序列，并将每个预 token 表示为 UTF-8 字节序列，就像在 BPE 训练中一样。我们将在每个预 token 内将这些字节合并为词汇元素，独立处理每个预 token（不跨预 token 边界合并）。

**步骤 2：应用合并。** 然后，我们取 BPE 训练期间创建的词汇元素合并序列，并按创建顺序将其应用于我们的预 token。

#### 示例（bpe_encoding）：BPE 编码示例

例如，假设我们的输入字符串是 `'the cat ate'`，我们的词汇表是 `{0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b'c', 8: b'a', 9: b'the', 10: b'at'}`，我们学到的合并是 `{(b't', b'h'), (b' ', b'c'), (b't', b'a'), (b'th', b'e'), (b'a', b't')}`。

---

## 第 11 页

1. 首先，我们的预分词器会将这个字符串分割为 `['the', 'cat', 'ate']`。然后，我们查看每个预 token 并应用 BPE 合并。

第一个预 token `'the'` 最初表示为 `[b't', b'h', b'e']`。查看我们的合并列表，我们识别出第一个适用的合并是 `(b't', b'h')`，用它把预 token 转换为 `[b'th', b'e']`。然后，我们回到合并列表，识别出下一个适用的合并是 `(b'th', b'e')`，将预 token 转换为 `[b'the']`。最后，再看合并列表，我们看到没有更多适用于该字符串的合并（因为整个预 token 已合并为单个 token），所以完成应用 BPE 合并。对应的整数序列是 `[9]`。

对剩余预 token 重复此过程，我们看到预 token `'cat'` 在应用 BPE 合并后表示为 `[b'c', b'a', b't']`，变成整数序列 `[7, 1, 5]`。最后的预 token `'ate'` 在应用 BPE 合并后是 `[b'at', b'e']`，变成整数序列 `[10, 3]`。因此，编码输入字符串的最终结果是 `[9, 7, 1, 5, 10, 3]`。

#### 特殊 token

你的分词器在编码文本时应能正确处理用户定义的特殊 token（构造分词器时提供）。

#### 内存考虑

假设我们想 token 化一个无法放入内存的大型文本文件。为了高效地 token 化这个大文件（或任何其他数据流），我们需要将其分成可管理的块并依次处理每个块，这样内存复杂度是常数，而不是随文本大小线性增长。这样做时，我们需要确保 token 不跨块边界，否则我们会得到与在内存中 token 化整个序列的朴素方法不同的分词结果。

#### 2.6.2 解码文本

要将整数 token ID 序列解码回原始文本，我们可以简单地查找每个 ID 在词汇表中对应的条目（字节序列），将它们拼接在一起，然后将字节解码为 Unicode 字符串。注意，输入 ID 不保证映射到有效的 Unicode 字符串（因为用户可以输入任何整数 ID 序列）。如果输入 token ID 不产生有效的 Unicode 字符串，你应该用官方 Unicode 替换字符 U+FFFD 替换格式错误的字节。`bytes.decode` 的 `errors` 参数控制 Unicode 解码错误如何处理，使用 `errors='replace'` 会自动用替换标记替换格式错误的数据。

#### 问题（tokenizer）：实现分词器（15 分）

交付物：实现一个 `Tokenizer` 类，给定词汇表和合并列表，将文本编码为整数 ID，并将整数 ID 解码为文本。你的分词器还应支持用户提供的特殊 token（如果尚未在词汇表中，则将它们追加到词汇表）。我们推荐以下接口：

---

## 第 12 页

```python
def __init__(self, vocab, merges, special_tokens=None)
```

从给定词汇表、合并列表和（可选）特殊 token 列表构造分词器。此函数应接受以下参数：

- `vocab: dict[int, bytes]`
- `merges: list[tuple[bytes, bytes]]`
- `special_tokens: list[str] | None = None`

```python
def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)
```

类方法，从序列化的词汇表和合并列表（格式与你的 BPE 训练代码输出相同）构造并返回 `Tokenizer`，并（可选）支持特殊 token。此方法应接受以下附加参数：

- `vocab_filepath: str`
- `merges_filepath: str`
- `special_tokens: list[str] | None = None`

```python
def encode(self, text: str) -> list[int]
```

将输入文本编码为 token ID 序列。

```python
def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]
```

给定字符串可迭代对象（例如 Python 文件句柄），返回一个惰性产生 token ID 的生成器。这对于无法直接加载到内存的大型文件的内存高效 token 化是必需的。

```python
def decode(self, ids: list[int]) -> str
```

将 token ID 序列解码为文本。

为了用我们提供的测试测试你的 `Tokenizer`，你首先需要实现测试适配器 `[adapters.get_tokenizer]`。然后运行 `uv run pytest tests/test_tokenizer.py`。你的实现应该能够通过所有测试。

#### 2.7 实验

#### 问题（tokenizer_experiments）：分词器实验（4 分）

(a) 从 TinyStories 和 OpenWebText 中采样 10 个文档。使用你之前训练的 TinyStories 和 OpenWebText 分词器（词汇表大小分别为 10K 和 32K），将这些采样文档编码为整数 ID。每个分词器的压缩率（字节/token）是多少？

交付物：一到两句话回答。

(b) 如果你用 TinyStories 分词器 token 化 OpenWebText 样本会发生什么？比较压缩率和/或定性描述发生了什么。

交付物：一到两句话回答。

(c) 估计你的分词器吞吐量（例如，字节/秒）。token 化 Pile 数据集（825GB 文本）需要多长时间？

交付物：一到两句话回答。

(d) 使用你的 TinyStories 和 OpenWebText 分词器，将各自的训练集和开发集编码为整数 token ID 序列。我们稍后将用这些来训练语言模型。我们建议将 token ID 序列化为 `uint16` 数据类型的 NumPy 数组。为什么 `uint16` 是合适的选择？

---

## 第 13 页

交付物：一到两句话回答。

### 3 Transformer 语言模型架构

语言模型接受一批整数 token ID 序列（即形状为 `(batch_size, sequence_length)` 的 `torch.Tensor`），并返回（批量的）词汇表上的归一化概率分布（即形状为 `(batch_size, sequence_length, vocab_size)` 的 PyTorch Tensor），其中预测分布是针对每个输入 token 的下一个词。训练语言模型时，我们使用这些下一个词预测来计算实际下一个词与预测下一个词之间的交叉熵损失。从语言模型生成文本时，我们取最后时间步（即序列中最后一项）的预测下一个词分布来生成序列中的下一个 token（例如，取概率最高的 token、从分布中采样等），将生成的 token 加入输入序列，然后重复。

在本部分作业中，你将从头构建这个 Transformer 语言模型。我们将从模型的高层描述开始，然后逐步详细介绍各个组件。

#### 3.1 Transformer LM

给定 token ID 序列，Transformer 语言模型使用输入嵌入将 token ID 转换为稠密向量，将嵌入 token 通过 `num_layers` 个 Transformer 块，然后应用学习的线性投影（“输出嵌入”或“LM head”）产生预测的下一个 token logits。示意图见图 1。

图 1：我们的 Transformer 语言模型概览。

图 2：一个 pre-norm Transformer 块。

---

## 第 14 页

1

#### Token 嵌入

在第一步中，Transformer 将（批量的）token ID 序列嵌入为包含 token 身份信息的向量序列（图 1 中的红色块）。

更具体地说，给定 token ID 序列，Transformer 语言模型使用 token 嵌入层产生向量序列。每个嵌入层接受形状为 `(batch_size, sequence_length)` 的整数张量，并产生形状为 `(batch_size, sequence_length, d_model)` 的向量序列。

#### Pre-norm Transformer 块

嵌入之后，激活值由几个结构相同的神经网络层处理。标准 decoder-only Transformer 语言模型由 `num_layers` 个相同的层（通常称为 Transformer “块”）组成。每个 Transformer 块接受形状为 `(batch_size, sequence_length, d_model)` 的输入，并返回形状为 `(batch_size, sequence_length, d_model)` 的输出。每个块通过自注意力聚合序列中的信息，并通过前馈层对其进行非线性变换。

经过 `num_layers` 个 Transformer 块后，我们将最终激活值转换为词汇表上的分布。

我们将实现“pre-norm”Transformer 块（详见 [第 3.4 节]），它额外要求在最后一个 Transformer 块之后使用层归一化（详见下文），以确保其输出被适当缩放。

在此归一化之后，我们将使用标准的学习线性变换将 Transformer 块的输出转换为预测的下一个 token logits（参见例如 [A. Radford et al. [7] 公式 2]）。

#### 3.2 备注：批处理、Einsum 和高效计算

在整个 Transformer 中，我们将对许多类似批次的输入执行相同计算。以下是一些例子：

- 批的元素：我们对每个批次元素应用相同的 Transformer 前向操作。
- 序列长度：“逐位置”操作（如 RMSNorm 和前馈）在序列的每个位置上完全相同地操作。
- 注意力头：注意力操作在“多头”注意力操作中跨注意力头批处理。

拥有一种符合人体工程学的方式来执行此类操作，以充分利用 GPU，并且易于阅读和理解，是很有用的。许多 PyTorch 操作可以在张量开头接受额外的“类似批次”维度，并高效地跨这些维度重复/广播操作。

例如，假设我们正在做逐位置的批量操作。我们有一个形状为 `(batch_size, sequence_length, d_model)` 的“数据张量”`D`，我们想对形状为 `(d_model, d_model)` 的矩阵 `A` 做批量向量-矩阵乘法。在这种情况下，`D @ A` 将执行批量矩阵乘法，这是 PyTorch 中高效的基元，其中 `(batch_size, sequence_length)` 维度被批处理。

因此，假设你的函数可能会被赋予额外的类似批次维度，并将这些维度保留在 PyTorch 形状的开头是有帮助的。为了以这种方式组织张量以便批处理，可能需要使用许多步骤的 `view`、`reshape` 和 `transpose`。这可能有点麻烦，而且常常难以阅读代码在做什么以及张量的形状是什么。

---

## 第 15 页

更符合人体工程学的选择是在 `torch.einsum` 中使用 einsum 表示法，或使用框架无关的库如 `einops` 或 `einx`。两个关键操作是 `einsum`，它可以对任意维度的输入张量进行张量收缩；以及 `rearrange`，它可以重新排序、连接和拆分任意维度。事实证明，机器学习中几乎所有操作都是维度调整和张量收缩的组合，偶尔加上（通常是逐点的）非线性函数。这意味着，当使用 einsum 表示法时，你的许多代码可以更易读、更灵活。

我们强烈建议在课程中学习并使用 einsum 表示法。以前没有接触过 einsum 表示法的学生应使用 `einops`（文档见此处），已经熟悉 `einops` 的学生应学习更通用的 `einx`（此处）。这两个包已安装在提供的环境中。

这里给出一些如何使用 einsum 表示法的例子。这些是对 `einops` 文档的补充，你应该先阅读该文档。

#### 示例（einstein_example1）：使用 einops.einsum 的批量矩阵乘法

```python
import torch
from einops import rearrange, einsum

## 基本实现
Y = D @ A.T

# 很难看出输入和输出形状以及它们的含义。
# D 和 A 可以有什么形状，这些是否有意外行为？

## Einsum 自文档化且稳健
# D A -> Y
Y = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")

## 或者，批量版本，其中 D 可以有任意前导维度，但 A 受约束。
Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
```

#### 示例（einstein_example2）：使用 einops.rearrange 的广播操作

我们有一批图像，对于每张图像，我们想根据某个缩放因子生成 10 个变暗版本：

```python
images = torch.randn(64, 128, 128, 3)  # (batch, height, width, channel)
dim_by = torch.linspace(start=0.0, end=1.0, steps=10)

## 重塑并相乘
dim_value = rearrange(dim_by, "dim_value -> 1 dim_value 1 1 1")
images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")
dimmed_images = images_rearr * dim_value

## 或者一次完成：
dimmed_images = einsum(
    images, dim_by,
    "batch height width channel, dim_value -> batch dim_value height width channel"
)
```

---

## 第 16 页

1

假设我们有一批图像，表示为形状为 `(batch, height, width, channel)` 的张量，我们想对图像的所有像素执行线性变换，但该变换应独立地对每个通道进行。我们的线性变换表示为形状为 `(height * width, height * width)` 的矩阵 `B`。

```python
channels_last = torch.randn(64, 32, 32, 3)  # batch, height, width, channel
B = torch.randn(32*32, 32*32)

## 重新排列图像张量以跨所有像素混合
channels_last_flat = channels_last.view(
    -1, channels_last.size(1) * channels_last.size(2), channels_last.size(3)
)
channels_first_flat = channels_last_flat.transpose(1, 2)
channels_first_flat_transformed = channels_first_flat @ B.T
channels_last_flat_transformed = channels_first_flat_transformed.transpose(1, 2)
channels_last_transformed = channels_last_flat_transformed.view(*channels_last.shape)
```

相反，使用 `einops`：

```python
height = width = 32
## Rearrange 替代笨拙的 torch view + transpose
channels_first = rearrange(
    channels_last,
    "batch height width channel -> batch channel (height width)"
)
channels_first_transformed = einsum(
    channels_first, B,
    "batch channel pixel_in, pixel_out pixel_in -> batch channel pixel_out"
)
channels_last_transformed = rearrange(
    channels_first_transformed,
    "batch channel (height width) -> batch height width channel",
    height=height, width=width
)
```

或者，如果你想更疯狂：使用 `einx.dot`（`einops.einsum` 的 einx 等价物）一次完成：

```python
height = width = 32
channels_last_transformed = einx.dot(
    "batch row_in col_in channel, (row_out col_out) (row_in col_in) -> batch row_out col_out channel",
    channels_last, B,
    col_in=width, col_out=width
)
```

这里第一个实现可以通过在前后放置注释来指示输入和输出形状来改进，但这很笨拙且容易出错。使用 einsum 表示法，文档就是实现！

Einsum 表示法可以处理任意输入批处理维度，但也有自文档化的关键好处。在使用 einsum 表示法的代码中，输入和输出张量的相关形状要清楚得多。

---

## 第 17 页

#### 3.2.1 数学符号和内存顺序

对于剩余张量，你可以考虑使用 Tensor 类型提示，例如使用 `jaxtyping` 库（不特定于 JAX）。

我们将在作业 2 中进一步讨论使用 einsum 表示法的性能影响，但现在要知道它们几乎总是比替代方案更好！

许多机器学习论文在符号中使用行向量，这导致表示与 NumPy 和 PyTorch 默认使用的行主内存顺序很好地吻合。使用行向量，线性变换看起来像

\[
y = xW^{\top}, \quad (1)
\]

对于行主 `W ∈ R^{d_out × d_in}` 和行向量 `x ∈ R^{1 × d_in}`。注意，这让我们可以通过增加 `x` 的最外维度来批处理输入，这意味着我们可以将向量输入 `x` 替换为矩阵输入 `X ∈ R^{batch × d_in}`。

在线性代数中，更常见的是使用列向量，其中线性变换看起来像

\[
y = Wx, \quad (2)
\]

给定行主 `W ∈ R^{d_out × d_in}` 和列向量 `x ∈ R^{d_in}`。在这种设置中批处理输入，批次维度必须放在 `x` 的最后，所以 `x` 需要替换为矩阵 `\tilde{X} ∈ R^{d_in × batch}`。

在本作业中，我们主要使用列向量进行数学符号表示，因为数学通常遵循这种符号。你应该记住，如果你想使用普通矩阵乘法符号，由于 PyTorch 使用行主内存顺序，你将不得不像方程 1 中的行向量约定那样对矩阵应用转置。如果你使用 einsum 进行线性代数操作，只要你正确标记轴，这应该不是问题。顺便提一下，值得注意的是，其他语言/线性代数包如 Matlab、Julia 和 Fortran 都使用列主内存顺序，这意味着批处理维度放在最后，但 Python 和相关包采用了 C 标准的行主顺序。

#### 3.3 基本构建块：Linear 和 Embedding 模块

#### 3.3.1 参数初始化

有效训练神经网络通常需要仔细初始化模型参数——糟糕的初始化可能导致不良行为，如梯度消失或爆炸。Pre-norm Transformer 对初始化异常稳健，但它们仍然可能对训练速度和收敛产生显著影响。由于本作业已经很长，我们将细节留到作业 3，而是给你一些近似初始化，应该适用于大多数情况。现在，使用：

- Linear 权重：\(\mathcal{N}\left(\mu = 0,\sigma^{2} = \frac{2}{d_{\mathrm{in}} + d_{\mathrm{out}}}\right)\) 截断在 \([- 3\sigma ,3\sigma ]\)
- Embedding：\(\mathcal{N}(\mu = 0,\sigma^{2} = 1)\) 截断在 \([- 3,3]\)
- RMSNorm：1

你应该使用 `torch.nn.init.trunc_normal_` 初始化截断正态权重。

#### 3.3.2 Linear 模块

线性层是 Transformer 和神经网络的基本构建块。首先，你将实现自己的 `Linear` 类，继承自 `torch.nn.Module` 并执行线性变换：

---

## 第 18 页

3

注意，我们不包括偏置项，遵循大多数现代 LLM。

#### 问题（linear）：实现线性模块（1 分）

交付物：实现一个 `Linear` 类，继承自 `torch.nn.Module` 并执行线性变换。你的实现应遵循 PyTorch 内置 `nn.Linear` 模块的接口，但没有偏置参数或参数。我们推荐以下接口：

```python
def __init__(self, in_features, out_features, device=None, dtype=None)
```

构造线性变换模块。此函数应接受以下参数：

- `in_features: int` 输入的最终维度
- `out_features: int` 输出的最终维度
- `device: torch.device | None = None` 存储参数的设备
- `dtype: torch.dtype | None = None` 参数的数据类型

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

将线性变换应用于输入。

确保：

- 子类化 `nn.Module`
- 调用超类构造函数
- 构造并存储参数为 `W`（不是 `W^T`），当然放在 `nn.Parameter` 中，不要使用 `nn.Linear` 或 `nn.functional.Linear`

对于初始化，使用上述设置以及 `torch.nn.init.trunc_normal_` 初始化权重。

为了测试你的 `Linear` 模块，实现测试适配器 `[adapters.run_linear]`。适配器应将给定权重加载到你的 `Linear` 模块中。你可以使用 `Module.load_state_dict` 实现此目的。然后运行 `uv run pytest -k test_linear`。

#### 3.3.3 Embedding 模块

如上所述，Transformer 的第一层是嵌入层，将整数 token ID 映射到维度为 `d_model` 的向量空间。我们将实现一个自定义 `Embedding` 类，继承自 `torch.nn.Module`（因此你不应使用 `nn.Embedding`）。`forward` 方法应通过使用形状为 `(batch_size, sequence_length)` 的 `torch.LongTensor` token ID 索引形状为 `(vocab_size, d_model)` 的嵌入矩阵，为每个 token ID 选择嵌入向量。

#### 问题（embedding）：实现嵌入模块（1 分）

交付物：实现 `Embedding` 类，继承自 `torch.nn.Module` 并执行嵌入查找。你的实现应遵循 PyTorch 内置 `nn.Embedding` 模块的接口。我们推荐以下接口：

---

## 第 19 页

```python
def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None)
```

构造嵌入模块。此函数应接受以下参数：

- `num_embeddings: int` 词汇表大小
- `embedding_dim: int` 嵌入向量维度，即 `d_model`
- `device: torch.device | None = None` 存储参数的设备
- `dtype: torch.dtype | None = None` 参数的数据类型

```python
def forward(self, token_ids: torch.Tensor) -> torch.Tensor
```

查找给定 token ID 的嵌入向量。

确保：

- 子类化 `nn.Module`
- 调用超类构造函数
- 将嵌入矩阵初始化为 `nn.Parameter`
- 存储嵌入矩阵，`d_model` 作为最终维度
- 当然，不要使用 `nn.Embedding` 或 `nn.functional_embedding`

同样，使用上述设置进行初始化，并使用 `torch.nn.init.trunc_normal_` 初始化权重。

为了测试你的实现，实现测试适配器 `[adapters.run_embedding]`。然后运行 `uv run pytest -k test_embedding`。

#### 3.4 Pre-Norm Transformer 块

每个 Transformer 块有两个子层：多头自注意力机制和逐位置前馈网络（[A. Vaswani et al., 2017]，第 3.1 节）。

在原始 Transformer 论文中，模型围绕两个子层使用残差连接，然后进行层归一化。这种架构通常称为“post-norm”Transformer，因为层归一化应用于子层输出。然而，许多工作发现，将层归一化从每个子层的输出移到每个子层的输入（并在最终 Transformer 块之后增加额外的层归一化）可以提高 Transformer 训练稳定性 [T. Q. Nguyen et al., 2019], [R. Xiong et al., 2020]——见图 2 中这种“pre-norm”Transformer 块的可视化表示。然后，每个 Transformer 块子层的输出通过残差连接加到子层输入上（[A. Vaswani et al., 8]，第 5.4 节）。Pre-norm 的直觉是，从输入嵌入到 Transformer 最终输出有一条干净的“残差流”，没有任何归一化，据称可以改善梯度流。这种 pre-norm Transformer 现在是当今语言模型（例如 GPT-3、LLaMA、PaLM 等）使用的标准，所以我们将实现这个变体。我们将逐一介绍 pre-norm Transformer 块的每个组件，按顺序实现它们。

#### 3.4.1 均方根层归一化

原始 Transformer 实现 [A. Vaswani et al., 8] 使用层归一化 [J. L. Ba et al., 2016] 来归一化激活。遵循 [H. Touvron et al., 12]，我们将使用均方根层归一化（RMSNorm；[B. Zhang et al., 13]，公式 4）进行层归一化。给定激活向量 \(a \in \mathbb{R}^{d_{\mathrm{model}}}\)，RMSNorm 将重新缩放每个激活 \(a_i\) 如下：

\[
\mathrm{RMSNorm}(a_i) = \frac{a_i}{\mathrm{RMS}(a)} g_i, \quad (4)
\]

---

## 第 20 页

1

其中 \(\mathrm{RMS}(a) = \sqrt{\frac{1}{d_{\mathrm{model}}}\sum_{i = 1}^{d_{\mathrm{model}}}a_i^2} +\epsilon\)。这里，\(g_{i}\) 是可学习的“增益”参数（总共有 d_model 个这样的参数），\(\epsilon\) 是超参数，通常固定为 1e-5。你应该将输入上推到 `torch.float32` 以防止平方输入时溢出。总体而言，你的 `forward` 方法应如下所示：

```python
in_dtype = x.dtype
x = x.to(torch.float32)

# 你的代码在这里执行 RMSNorm
result = ...

# 以原始 dtype 返回结果
return result.to(in_dtype)
```

#### 问题（rmsnorm）：均方根层归一化（1 分）

交付物：实现 `RMSNorm` 作为 `torch.nn.Module`。我们推荐以下接口：

```python
def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)
```

构造 `RMSNorm` 模块。此函数应接受以下参数：

- `d_model: int` 模型的隐藏维度
- `eps: float = 1e-5` 用于数值稳定性的 epsilon 值
- `device: torch.device | None = None` 存储参数的设备
- `dtype: torch.dtype | None = None` 参数的数据类型

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

处理形状为 `(batch_size, sequence_length, d_model)` 的输入张量，并返回相同形状的张量。

注意：记住在执行归一化之前将输入上推到 `torch.float32`（然后下推到原始 dtype），如上所述。

为了测试你的实现，实现测试适配器 `[adapters.run_rmsnorm]`。然后运行 `uv run pytest -k test_rmsnorm`。

---

## 第 21 页

#### 3.4.2 逐位置前馈网络

图 3：比较 SiLU（又名 Swish）和 ReLU 激活函数。

在原始 Transformer 论文（[A. Vaswani et al. [8] 第 3.3 节）中，Transformer 前馈网络由两个线性变换组成，中间有 ReLU 激活（\(\mathrm{ReLU}(x) = \max (0,x)\)）。在原始架构中，内部前馈层的维度通常是输入维度的 4 倍。

然而，现代语言模型倾向于与此原始设计相比有两个主要变化：它们使用另一种激活函数，并采用门控机制。具体来说，我们将实现 LLM（如 Llama 3 [A. Grattafiori et al., 2024] 和 Qwen 2.5 [A. Yang et al., 2024]）采用的“SwiGLU”激活函数，它将 SiLU（通常称为 Swish）激活与称为门控线性单元（GLU）的门控机制相结合。我们还将省略线性层中有时使用的偏置项，遵循大多数现代 LLM（自 PaLM [A. Chowdhery et al., 2022] 和 LLaMA [H. Touvron et al., 2023] 以来）。

SiLU 或 Swish 激活函数 [D. Hendrycks et al., 2016; S. Elfwing et al., 2017] 定义如下：

\[
\mathrm{SiLU}(x) = x\cdot \sigma (x) = \frac{x}{1 + e^{-x}} \quad (5)
\]

如图 3 所示，SiLU 激活函数类似于 ReLU 激活函数，但在零处是平滑的。

门控线性单元（GLU）最初由 [Y. N. Dauphin et al. [19] 定义为通过 sigmoid 函数的线性变换与另一个线性变换的逐元素乘积：

\[
\mathrm{GLU}(x,W_1,W_2) = \sigma (W_1x)\odot W_2x, \quad (6)
\]

其中 \(\odot\) 表示逐元素乘法。门控线性单元被认为可以“通过为梯度提供线性路径，同时保留非线性能力，减少深度架构的梯度消失问题”。

---

## 第 22 页

1

\[
\mathrm{FFN}(x) = \mathrm{SwiGLU}(x,W_1,W_2,W_3) = W_2(\mathrm{SiLU}(W_1x)\odot W_3x), \quad (7)
\]

其中 \(x\in \mathbb{R}^{d_{\mathrm{model}}}\)，\(W_{1},W_{3}\in \mathbb{R}^{d_{\mathrm{in}}\times d_{\mathrm{model}}}\)，\(W_{2}\in \mathbb{R}^{d_{\mathrm{model}}\times d_{\mathrm{in}}}\)，并且通常 \(d_{\mathrm{in}} = \frac{8}{3} d_{\mathrm{model}}\)。在具体实现中，为了硬件效率，可以将其舍入到附近的 64 的倍数。

N. Shazeer [20] 首先提出将 SiLU/Swish 激活与 GLU 结合，并进行了实验，表明 SwiGLU 在语言建模任务上优于 ReLU 和 SiLU（无门控）等基线。在作业后面，你将比较 SwiGLU 和 SiLU。尽管我们已经提到这些组件的一些启发式论据（论文提供了更多支持证据），但保持实证视角是好的：Shazeer 论文中一句现在著名的话是

“我们无法解释为什么这些架构似乎有效；我们将它们的成功，如同其他一切，归因于神圣的仁慈。”

#### 问题（positionwise_feedforward）：实现逐位置前馈网络（2 分）

交付物：实现 SwiGLU 前馈网络，由 SiLU 激活函数和 GLU 组成。

注意：在这种特殊情况下，为了数值稳定性，你可以随意在实现中使用 `torch.sigmoid`。

你应该在实现中将 \(d_{\mathrm{ff}}\) 设置为大约 \(\frac{8}{3}\times d_{\mathrm{model}}\)，同时确保内部前馈层的维度是 64 的倍数，以充分利用硬件。为了用我们提供的测试测试你的实现，你需要实现测试适配器 `[adapters.run_swiglu]`。然后运行 `uv run pytest -k test_swiglu` 测试你的实现。

#### 3.4.3 相对位置嵌入

为了将位置信息注入模型，我们将实现旋转位置嵌入 [J. Su et al., 2021]，通常称为 RoPE。对于给定查询 token \(q^{(i)} = W_{q}x^{(i)}\in \mathbb{R}^{d}\) 在 token 位置 \(i\)，我们将应用成对旋转矩阵 \(R^{i}\)，得到 \(q^{\prime (i)} = R^{i}q^{(i)} = R^{i}W_{q}x^{(i)}\)。这里，\(R^{i}\) 将成对的嵌入元素 \(q_{2k - 1,2k}^{(i)}\) 作为 2D 向量旋转角度 \(\theta_{i,k} = \frac{1}{\Theta^{2k - 2 / d}}\)，其中 \(k\in \{1,\dots,d / 2\}\)，\(\Theta\) 是某个常数。因此，我们可以认为 \(R^{i}\) 是大小为 \(d\times d\) 的块对角矩阵，块为 \(R_{k}^{i}\)，\(k\in \{1,\dots,\frac{d}{2}\}\)，其中

\[
R_{k}^{i} = \left( \begin{array}{cc}\cos (\theta_{i,k}) & -\sin (\theta_{i,k})\\ \sin (\theta_{i,k}) & \cos (\theta_{i,k}) \end{array} \right) \quad (8)
\]

因此我们得到完整旋转矩阵

---

## 第 23 页

9

\[
R^{i} = \left( \begin{array}{ccccc}R_{1}^{i} & 0 & 0 & \dots & 0\\ 0 & R_{2}^{i} & 0 & \dots & 0\\ 0 & 0 & R_{3}^{i} & \dots & 0\\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \dots & R_{d / 2}^{i} \end{array} \right), \quad (9)
\]

其中 0 表示 \(2\times 2\) 零矩阵。虽然可以构造完整的 \(d\times d\) 矩阵，但好的解决方案应利用该矩阵的性质来更高效地实现变换。由于我们只关心给定序列内 token 的相对旋转，我们可以重用为 \(\cos (\theta_{i,k})\) 和 \(\sin (\theta_{i,k})\) 计算的值，跨层和不同批次。如果你想优化它，可以使用一个被所有层引用的单个 RoPE 模块，并且它可以在初始化时用 `self.register_buffer(persistent=False)` 创建一个 2D 预计算的 sin 和 cos 值缓冲区，而不是 `nn.Parameter`（因为我们不想学习这些固定的余弦和正弦值）。然后对 \(k^{(j)}\) 执行与我们对 \(q^{(i)}\) 所做的完全相同的旋转过程，用对应的 \(R^{j}\) 旋转。注意该层没有可学习参数。

#### 问题（rope）：实现 RoPE（2 分）

交付物：实现一个类 `RotaryPositionalEmbedding`，将 RoPE 应用于输入张量。

推荐以下接口：

```python
def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None)
```

构造 RoPE 模块并根据需要创建缓冲区。

- `theta: float` RoPE 的 \(\Theta\) 值
- `d_k: int` 查询和键向量的维度
- `max_seq_len: int` 将输入的最大序列长度
- `device: torch.device | None = None` 存储缓冲区的设备

```python
def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor
```

处理形状为 `(..., seq_len, d_k)` 的输入张量，并返回相同形状的张量。注意，你应该容忍 `x` 具有任意数量的批次维度。你应该假设 token 位置是形状为 `(..., seq_len)` 的张量，指定 `x` 沿序列维度的 token 位置。

你应该使用 token 位置来沿序列维度切片你的（可能是预计算的）cos 和 sin 张量。

为了测试你的实现，完成 `[adapters.run_rope]` 并确保它通过 `uv run pytest -k test_rope`。

#### 3.4.4 缩放点积注意力

我们现在将实现 [A. Vaswani et al. [8]（第 3.2.1 节）中描述的缩放点积注意力。作为预备步骤，注意力操作的定义将使用 softmax，该操作将未归一化的分数向量转换为归一化分布：

\[
\mathrm{softmax}(v)_i = \frac{\exp(v_i)}{\sum_{j = 1}^{n}\exp(v_j)}. \quad (10)
\]

---

## 第 24 页

注意，对于大值，\(\exp (v_{i})\) 可能变为 inf（然后，\(\inf_{i = 1}^{n} = \mathrm{NaN}\)）。我们可以通过注意到 softmax 操作对给所有输入加上任意常数 \(c\) 不变来避免这一点。我们可以利用这一性质提高数值稳定性——通常，我们会从 \(v\) 的所有元素中减去 \(v\) 的最大项，使新最大项为 0。你现在将实现 softmax，使用这个技巧提高数值稳定性。

#### 问题（softmax）：实现 softmax（1 分）

交付物：编写一个函数，对张量应用 softmax 操作。你的函数应接受两个参数：一个张量和一个维度 \(i\)，并对输入张量的第 \(i\) 维应用 softmax。输出张量应与输入张量形状相同，但其第 \(i\) 维现在具有归一化概率分布。使用从第 \(i\) 维所有元素中减去第 \(i\) 维最大值以避免数值稳定性问题的技巧。

为了测试你的实现，完成 `[adapters.run_softmax]` 并确保它通过 `uv run pytest -k test_softmax_matches_pytorch`。

我们现在可以数学上定义注意力操作如下：

\[
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \quad (11)
\]

其中 \(Q\in \mathbb{R}^{n\times d_k}\)，\(K\in \mathbb{R}^{m\times d_k}\)，\(V\in \mathbb{R}^{m\times d_v}\)。这里，\(Q\)、\(K\) 和 \(V\) 都是该操作的输入——注意这些不是可学习参数。

**掩码：** 有时方便地掩盖注意力操作的输出。掩码应具有形状 \(M\in \{\mathrm{True},\mathrm{False}\}^{n\times m}\)，其每一行 \(i\) 表示查询 \(i\) 应关注哪些键。典型地（且稍微令人困惑），位置 \((i,j)\) 处的值为 `True` 表示查询 \(i\) 确实关注键 \(j\)，值为 `False` 表示查询不关注键。换句话说，信息在值为 `True` 的 \((i,j)\) 对处“流动”。例如，考虑一个 \(1\times 3\) 掩码矩阵，条目为 `[[True, True, False]]`。单个查询向量只关注前两个键。

在计算上，使用掩码比在子序列上计算注意力要高效得多，我们可以通过取 softmax 前的值 \(\left(QK^{\top} / \sqrt{d_k}\right)\) 并向掩码矩阵中任何为 `False` 的条目添加 \(- \infty\) 来实现。

#### 问题（scaled_dot_product_attention）：实现缩放点积注意力（5 分）

交付物：实现缩放点积注意力函数。你的实现应处理形状为 `(batch_size, ..., seq_len, d_k)` 的键和查询，以及形状为 `(batch_size, ..., seq_len, d_v)` 的值，其中 `...` 表示任意数量的其他类似批次维度（如果提供）。实现应返回形状为 `(batch_size, ..., seq_len, d_v)` 的输出。关于类似批次维度的讨论见第 3.2 节。

你的实现还应支持可选的用户提供的形状为 `(seq_len, seq_len)` 的布尔掩码。掩码值为 `True` 的位置的注意力概率应总和为 1，掩码值为 `False` 的位置的注意力概率应为零。

---

## 第 25 页

为了用我们提供的测试测试你的实现，你需要实现测试适配器 `[adapters.run_scaled_dot_product_attention]`。`uv run pytest -k test_scaled_dot_product_attention` 测试你的实现在三阶输入张量上的表现，而 `uv run pytest -k test_4d_scaled_dot_product_attention` 测试你的实现在四阶输入张量上的表现。

#### 3.4.5 因果多头自注意力

我们将实现 [A. Vaswani et al. [8] 第 3.2.2 节中描述的多头自注意力。回想一下，数学上，应用多头注意力的操作定义如下：

\[
\mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h) \quad (12)
\]

\[
\mathrm{for~head}_i = \mathrm{Attention}(Q_i,K_i,V_i) \quad (13)
\]

其中 \(Q_{i}\)、\(K_{i}\)、\(V_{i}\) 是 \(Q\)、\(K\)、\(V\) 的嵌入维度中大小为 \(d_{k}\) 或 \(d_{v}\) 的第 \(i \in \{1, \ldots , h\}\) 个切片。Attention 是第 3.4.4 节定义的缩放点积注意力操作。由此我们可以形成多头自注意力操作：

\[
\mathrm{MultiHeadSelfAttention}(x) = W_O\mathrm{MultiHead}(W_Qx,W_Kx,W_Vx) \quad (14)
\]

这里，可学习参数是 \(W_{Q} \in \mathbb{R}^{h d_{k} \times d_{\mathrm{model}}}\)，\(W_{K} \in \mathbb{R}^{h d_{k} \times d_{\mathrm{model}}}\)，\(W_{V} \in \mathbb{R}^{h d_{v} \times d_{\mathrm{model}}}\)，以及 \(W_{O} \in \mathbb{R}^{d_{\mathrm{model}} \times h d_{v}}\)。由于 \(Q_{s}\)、\(K_{s}\)、\(V_{s}\) 在多头部注意力操作中被切片，我们可以认为 \(W_{Q}\)、\(W_{K}\) 和 \(W_{V}\) 沿输出维度为每个头分开。当你实现这一点时，你应该总共用三次矩阵乘法计算键、值和查询投影。\(^{5}\)

#### 因果掩码

你的实现应防止模型关注序列中的未来 token。换句话说，如果模型给定 token 序列 \(t_{1}, \ldots , t_{n}\)，并且我们想计算前缀 \(t_{1}, \ldots , t_{i}\)（其中 \(i < n\)）的下一个词预测，模型不应能够访问（关注）位置 \(t_{i + 1}, \ldots , t_{n}\) 的 token 表示，因为它在推理生成文本时不会访问这些 token（并且这些未来 token 会泄露真实下一个词身份的信息，使语言建模预训练目标变得微不足道）。对于输入 token 序列 \(t_{1}, \ldots , t_{n}\)，我们可以天真地通过运行多头自注意力 \(n\) 次（针对序列中的 \(n\) 个唯一前缀）来防止访问未来 token。相反，我们将使用因果注意力掩码，它允许 token \(i\) 关注序列中所有位置 \(j \leq i\)。你可以使用 `torch.triu` 或广播索引比较来构造这个掩码，并且你应该利用第 3.4.4 节的缩放点积注意力实现已经支持注意力掩码这一事实。

#### 应用 RoPE

RoPE 应应用于查询和键向量，但不应用于值向量。此外，头维度应作为批次维度处理，因为在多头注意力中，注意力是独立地对每个头应用的。这意味着，对于每个头，应精确地对查询和键向量应用相同的 RoPE 旋转。

#### 问题（multihead_self_attention）：实现因果多头自注意力（5 分）

---

## 第 26 页

1

交付物：实现因果多头自注意力作为 `torch.nn.Module`。你的实现应至少接受以下参数：

- `d_model: int` Transformer 块输入的维度。
- `num_heads: int` 多头自注意力中使用的头数。

遵循 [A. Vaswani et al. [8]，设置 \(d_{k} = d_{v} = \frac{d_{\mathrm{model}}}{h}\)。为了用我们提供的测试测试你的实现，实现测试适配器 `[adapters.run_multithread_self_attention]`。然后运行 `uv run pytest -k test_multithread_self_attention` 测试你的实现。

#### 3.5 完整 Transformer LM

让我们从组装 Transformer 块开始（参考图 2 会有帮助）。一个 Transformer 块包含两个“子层”，一个用于多头自注意力，另一个用于 SwiGLU 前馈网络。在每个子层中，我们首先执行 RMSNorm，然后执行主要操作（MHA/FF），最后加入残差连接。

具体来说，Transformer 块的第一半（第一个“子层”）应实现以下更新，从输入 \(x\) 产生输出 \(y\)：

\[
y = x + \mathrm{MultiHeadSelfAttention}(\mathrm{RMSNorm}(x)). \quad (15)
\]

#### 问题（transformer_block）：实现 Transformer 块（3 分）

实现第 3.4 节所述、图 2 所示的 pre-norm Transformer 块。你的 Transformer 块应至少接受以下参数：

- `d_model: int` Transformer 块输入的维度。
- `num_heads: int` 多头自注意力中使用的头数。
- `d_ff: int` 逐位置前馈内部层的维度。

为了测试你的实现，实现适配器 `[adapters.run_transformer_block]`。然后运行 `uv run pytest -k test_transformer_block` 测试你的实现。

交付物：通过提供测试的 Transformer 块代码。

现在我们把块组合起来，遵循图 1 的高层图。按照第 3.1.0.1 节对嵌入的描述，将其输入 `num_layers` 个 Transformer 块，然后将其传入最终层归一化和 LM head，以获得词汇表上的未归一化分布（logits）。

#### 问题（transformer_lm）：实现 Transformer LM（3 分）

是时候把它们全部组合起来了！实现第 3.1 节所述、图 1 所示的 Transformer 语言模型。至少，你的实现应接受前述 Transformer 块的所有构造参数，以及这些附加参数：

- `vocab_size: int` 词汇表大小，用于确定 token 嵌入矩阵的维度。

---

## 第 27 页

- `context_length: int` 最大上下文长度，用于确定 RoPE sin 和 cos 缓冲区的维度。
- `num_layers: int` 要使用的 Transformer 块数。

为了用我们提供的测试测试你的实现，你首先需要实现测试适配器 `[adapters.run_transformer_lm]`。然后运行 `uv run pytest -k test_transformer_lm` 测试你的实现。

交付物：通过上述测试的 Transformer LM 模块。

#### 资源核算

能够理解 Transformer 各部分如何消耗计算和内存是有用的。我们将逐步进行一些基本的“FLOPs 核算”。Transformer 中绝大多数 FLOPS 是矩阵乘法，所以我们的核心方法很简单：

1. 写下 Transformer 前向传播中的所有矩阵乘法。
2. 将每个矩阵乘法转换为所需的 FLOPs。

对于第二步，以下事实很有用：

**规则：** 给定 \(A \in \mathbb{R}^{m \times n}\) 和 \(B \in \mathbb{R}^{n \times p}\)，矩阵-矩阵乘积 \(AB\) 需要 \(2mnp\) FLOPs。

要看到这一点，注意 \((AB)[i,j] = A[i,:]\cdot B[:,j]\)，而这个点积需要 \(n\) 次加法和 \(n\) 次乘法（\(2n\) FLOPs）。然后，由于矩阵-矩阵乘积 \(AB\) 有 \(m\times p\) 个条目，总 FLOPs 数为 \((2n)(mp) = 2mnp\)。

现在，在你做下一个问题之前，逐步检查你的 Transformer 块和 Transformer LM 的每个组件，列出所有矩阵乘法及其相关 FLOPs 成本可能会有帮助。

#### 问题（transformer_accounting）：Transformer LM 资源核算（5 分）

(a) 考虑一个使用我们作业架构的 GPT-2 XL 规模模型，具有以下配置：

- `vocab_size`: 50,257
- `context_length`: 1,024
- `num_layers`: 48
- `d_model`: 1,600
- `num_heads`: 25
- `d_ff`: 4,288（最接近 \(\frac{8}{3}\times 1,600\) 的 64 的倍数）

假设我们用此配置构造模型。我们的模型有多少可训练参数？假设每个参数用单精度浮点表示，仅加载此模型需要多少内存？

交付物：一到两句话回答。

(b) 识别完成我们 GPT-2 XL 形状模型前向传播所需的矩阵乘法。这些矩阵乘法总共需要多少 FLOPs？假设我们的输入序列有 `context_length` 个 token。

---

## 第 28 页

1

交付物：矩阵乘法列表（带描述）以及所需总 FLOPs 数。

(c) 基于上述分析，模型的哪些部分需要最多 FLOPs？

交付物：一到两句话回答。

(d) 对 GPT-2 small（12 层，768 d_model，12 头）、GPT-2 medium（24 层，1024 d_model，16 头）和 GPT-2 large（36 层，1280 d_model，20 头）重复你的分析。随着模型规模增加，Transformer LM 的哪些部分在总 FLOPs 中占比更大或更小？

交付物：对于每个模型，提供模型组件分解及其相关 FLOPs（作为前向传播所需总 FLOPs 的比例）。此外，用一到两句话描述改变模型大小如何改变每个组件的比例 FLOPs。

(e) 取 GPT-2 XL 并将上下文长度增加到 16,384。一次前向传播的总 FLOPs 如何变化？模型组件的相对 FLOPs 贡献如何变化？

交付物：一到两句话回答。

### 4 训练 Transformer LM

我们现在有了预处理数据（通过分词器）和模型（Transformer）的步骤。剩下的是构建支持训练的所有代码。这包括以下内容：

- 损失：我们需要定义损失函数（交叉熵）。
- 优化器：我们需要定义优化器来最小化此损失（AdamW）。
- 训练循环：我们需要所有支持基础设施，加载数据、保存检查点和管理训练。

#### 4.1 交叉熵损失

回想一下，Transformer 语言模型为每个长度为 \(m + 1\) 的序列 \(x\) 和 \(i = 1, \ldots , m\) 定义分布 \(p_{\theta}(x_{i + 1} \mid x_{1:i})\)。给定由长度为 \(m + 1\) 的序列组成的训练集 \(D\)，我们定义标准交叉熵（负对数似然）损失函数：

\[
\ell (\theta ;D) = \frac{1}{|D|} m\sum_{x\in D}\sum_{i = 1}^{m} - \log p_{\theta}(x_{i + 1}\mid x_{1:i}). \quad (16)
\]

（注意，Transformer 的一次前向传播会为所有 \(i = 1, \ldots , m\) 产生 \(p_{\theta}(x_{i + 1} \mid x_{1:i})\)。）

特别地，Transformer 为每个位置 \(i\) 计算 logits \(o_{i} \in \mathbb{R}^{\mathrm{vocab\_size}}\)，这导致：

\[
p(x_{i + 1} \mid x_{1:i}) = \mathrm{softmax}(o_i)[x_{i + 1}] = \frac{\exp(o_i[x_{i + 1}])}{\sum_{a = 1}^{\mathrm{vocab\_size}}\exp(o_i[a])}. \quad (17)
\]

交叉熵损失通常针对 logits 向量 \(o_{i} \in \mathbb{R}^{\mathrm{vocab\_size}}\) 和目标 \(x_{i + 1}\) 定义。\(^{7}\)

---

## 第 29 页

实现交叉熵损失需要注意数值问题，就像 softmax 一样。

#### 问题（cross_entropy）：实现交叉熵（1 分）

交付物：编写一个函数来计算交叉熵损失，它接受预测 logits \((o_{i})\) 和目标 \((x_{i + 1})\)，并计算交叉熵 \(\ell_{i} = - \log \operatorname {softmax}(o_{i})[x_{i + 1}]\)。你的函数应处理以下内容：

- 减去最大元素以保证数值稳定性。
- 尽可能抵消 log 和 exp。
- 处理任何附加批次维度，并返回批次上的平均值。如第 3.2 节，我们假设类似批次维度总是在词汇表大小维度之前。

实现 `[adapters.run_cross_entropy]`，然后运行 `uv run pytest -k test_cross_entropy` 测试你的实现。

#### 困惑度

交叉熵足以用于训练，但当我们评估模型时，我们还希望报告困惑度。对于长度为 \(m\) 的序列，其交叉熵损失为 \(\ell_{1},\ldots ,\ell_{m}\)：

\[
\mathrm{perplexity} = \exp \left(\frac{1}{m}\sum_{i = 1}^{m}\ell_{i}\right). \quad (18)
\]

#### 4.2 SGD 优化器

现在我们有损失函数，我们将开始探索优化器。最简单的基于梯度的优化器是随机梯度下降（SGD）。我们从随机初始化的参数 \(\theta_{0}\) 开始。然后对于每一步 \(t = 0,\ldots ,T - 1\)，我们执行以下更新：

\[
\theta_{t + 1}\leftarrow \theta_{t} - \alpha_{t}\nabla L(\theta_{t};B_{t}), \quad (19)
\]

其中 \(B_{t}\) 是从数据集 \(D\) 中采样的随机数据批次，学习率 \(\alpha_{t}\) 和批次大小 \(|B_{t}|\) 是超参数。

#### 4.2.1 在 PyTorch 中实现 SGD

为了实现优化器，我们将子类化 PyTorch 的 `torch.optim.Optimizer` 类。`Optimizer` 子类必须实现两个方法：

- `__init__(self, params, ...)` 应初始化你的优化器。这里，`params` 是待优化参数的集合（或参数组，如果用户想对不同模型部分使用不同超参数，例如学习率）。确保将 `params` 传递给基类的 `__init__` 方法，它将存储这些参数供 `step` 使用。你可以根据优化器接受额外参数（例如学习率是常见的），并将它们作为字典传递给基类构造函数，键是你为这些参数选择的名称（字符串）。

- `step(self)` 应执行一次参数更新。在训练循环中，它将在反向传播后调用，因此你可以访问上一批次的梯度。此方法应遍历每个参数张量 \(\mathfrak{p}\) 并原地修改它们，即设置 `p.data`，它保存与该参数相关的张量，基于梯度 `p.grad`（如果存在），后者表示损失对该参数的梯度。

PyTorch 优化器 API 有一些微妙之处，所以用示例解释更容易。为了让示例更丰富，我们将实现 SGD 的一个变体，其中学习率随训练衰减，从初始学习率 \(\alpha\) 开始，并随时间采取越来越小的步长：

\[
\theta_{t + 1} = \theta_{t} - \frac{\alpha}{\sqrt{t + 1}}\nabla L(\theta_{t};B_{t}) \quad (20)
\]

让我们看看这个版本的 SGD 如何实现为 PyTorch Optimizer：

```python
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # 获取学习率。
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # 获取与 p 关联的状态。
                t = state.get("t", 0)  # 从状态中获取迭代次数，或 0。
                grad = p.grad.data  # 获取损失对 p 的梯度。
                p.data -= lr / math.sqrt(t + 1) * grad  # 原地更新权重张量。
                state["t"] = t + 1  # 增加迭代次数。

        return loss
```

在 `__init__` 中，我们将参数以及默认超参数传递给基类构造函数（参数可能成组出现，每组有不同的超参数）。如果参数只是单个 `torch.nn.Parameter` 对象集合，基类构造函数将创建一个组并分配默认超参数。然后，在 `step` 中，我们遍历每个参数组，然后遍历该组中的每个参数，并应用方程 20。这里，我们将迭代次数作为与每个参数关联的状态保存：我们先读取该值，在梯度更新中使用它，然后更新它。API 规定用户可能传入可调用 `closure` 以在优化器步骤前重新计算损失。我们不需要为我们将使用的优化器这样做，但为了符合 API，我们添加了它。

为了看到它工作，我们可以使用以下最小训练循环示例：

```python
weights = torch.nn.Parameter(5 * torch.randn(10, 10))
opt = SGD([weights], lr=1)

for t in range(100):
    opt.zero_grad()  # 重置所有可学习参数的梯度。
    loss = (weights**2).mean()  # 计算标量损失值。
    print(loss.cpu().item())
    loss.backward()  # 运行反向传播，计算梯度。
    opt.step()  # 运行优化器步骤。
```

---

## 第 31 页

这是训练循环的典型结构：在每次迭代中，我们将计算损失并运行优化器的一步。训练语言模型时，我们的可学习参数将来自模型（在 PyTorch 中，`m.parameters()` 给我们这个集合）。损失将在采样的数据批次上计算，但训练循环的基本结构相同。

#### 问题（learning_rate_tuning）：调整学习率（1 分）

正如我们将看到的，影响训练最大的超参数之一是学习率。让我们在上面的玩具示例中实际看看。用另外三个学习率值运行上面的 SGD 示例：`1e1`、`1e2` 和 `1e3`，仅训练 10 次迭代。每个学习率的损失会发生什么？它是衰减更快、更慢，还是发散（即训练过程中增加）？

交付物：一到两句话回答你观察到的行为。

#### 4.3 AdamW

现代语言模型通常使用更复杂的优化器训练，而不是 SGD。最近使用的大多数优化器都是 Adam 优化器 [D. P. Kingma et al., 2015] 的衍生物。我们将使用 AdamW [I. Loshchilov et al., 2019]，它在最近工作中广泛使用。AdamW 提出了对 Adam 的修改，通过添加权重衰减改善正则化（在每次迭代中，我们将参数拉向 0），以与梯度更新解耦的方式。我们将按照 [I. Loshchilov et al. [23]] 的算法 2 实现 AdamW。

AdamW 对每个参数是有状态的，它跟踪其第一和第二矩的运行估计。因此，AdamW 使用额外内存来换取更好的稳定性和收敛。除了学习率 \(\alpha\)，AdamW 还有一对超参数 \((\beta_{1}, \beta_{2})\) 控制矩估计的更新，以及权重衰减率 \(\lambda\)。典型应用将 \((\beta_{1}, \beta_{2})\) 设为 \((0.9, 0.999)\)，但大型语言模型如 LLaMA [H. Touvron et al., 2023] 和 GPT-3 [T. B. Brown et al., 2020] 通常用 \((0.9, 0.95)\) 训练。算法可以写成如下，其中 \(\epsilon\) 是一个小值（例如 \(10^{- 8}\)），用于在 \(v\) 中遇到极小值时提高数值稳定性：

---

## 第 32 页

1. `init(0)` 初始化可学习参数
2. \(m\gets 0\) 第一矩向量的初始值；形状与 \(\theta\) 相同
3. \(v\gets 0\) 第二矩向量的初始值；形状与 \(\theta\) 相同
4. for \(t = 1,\dots,T\) do
5. 采样数据批次 \(B_{t}\)
6. \(g\leftarrow \nabla_{\theta}\ell (\theta ;B_{t})\) 计算损失的梯度
7. \(\alpha_{t}\leftarrow \alpha \frac{\sqrt{1 - \beta_{t}^{2}}}{1 - \beta_{t}^{2}}\) 计算第 \(t\) 次迭代调整后的 \(\alpha\)
8. \(\theta \gets \theta -\alpha \lambda \theta\) 应用权重衰减
9. \(m\leftarrow \beta_{1}m + (1 - \beta_{1})g\) 更新第一矩估计
10. \(v\leftarrow \beta_{2}v + (1 - \beta_{2})g^{2}\) 更新第二矩估计
11. \(\theta \gets \theta -\alpha_{t}\frac{m}{\sqrt{v + \epsilon}}\) 应用矩调整后的权重更新
12. end for

算法 1：AdamW 优化器伪代码。

注意 \(t\) 从 1 开始。你现在将实现这个优化器。

#### 问题（adamw）：实现 AdamW（2 分）

交付物：实现 AdamW 优化器作为 `torch.optim.Optimizer` 的子类。你的类应在 `__init__` 中接受学习率 \(\alpha\)，以及 \(\beta\)、\(\epsilon\) 和 \(\lambda\) 超参数。为了帮助你保存状态，基类 `Optimizer` 类给你一个字典 `self.state`，它将 `nn.Parameter` 对象映射到存储该参数所需任何信息的字典（对于 AdamW，这将是矩估计）。实现 `[adapters.get_adamw_cls]` 并确保它通过 `uv run pytest -k test_adamw`。

#### 问题（adamw_accounting）：使用 AdamW 训练的资源核算（2 分）

让我们计算运行 AdamW 需要多少内存和计算。假设我们对每个张量使用 float32。

(a) 运行 AdamW 需要多少峰值内存？根据参数、激活、梯度和优化器状态的内存使用分解你的答案。用 `batch_size` 和模型超参数（`vocab_size`、`context_length`、`num_layers`、`d_model`、`num_heads`）表示答案。假设 \(d_{\mathrm{ff}} = \frac{8}{3} \times d_{\mathrm{model}}\)。

为简单起见，计算激活内存使用时，仅考虑以下组件：

- Transformer 块 RMSNorm(s)
- 多头自注意力子层：\(QKV\) 投影、\(QK^{\top}\) 矩阵乘法、softmax、值的加权和、输出投影。
- 逐位置前馈（SwiGLU）：\(W_{1}\)、\(W_{2}\)、门分支上的 SiLU、逐元素乘积、\(W_{3}\)

---

## 第 33 页

1. 最终 RMSNorm
2. 输出嵌入
3. logits 上的交叉熵

交付物：参数、激活、梯度和优化器状态各自的代数表达式，以及总量。

(b) 对你的答案实例化 GPT-2 XL 形状模型，得到仅依赖 `batch_size` 的表达式。在 80GB 内存内，你可以使用的最大批次大小是多少？

交付物：一个形如 \(a\cdot \mathrm{batch\_size} + b\) 的表达式，带有数值 \(a,b\)，以及表示最大批次大小的数字。

(c) 运行一步 AdamW 需要多少 FLOPs？

交付物：一个代数表达式，并简要说明理由。

(d) 模型 FLOPs 利用率（MFU）定义为观察到的吞吐量（token/秒）与硬件理论峰值 FLOP 吞吐量之比 [A. Chowdhery et al., 2022]。NVIDIA H100 GPU 对“float32”（实际上是 TensorFloat-32，现实中是“bfloat19”）操作的理论峰值为 495 teraFLOP/s。假设你能获得 50% MFU，在单个 H100 上训练 GPT-2 XL 400K 步、批次大小 1024 需要多长时间？遵循 [J. Kaplan et al. [25]] 和 [J. Hoffmann et al. [26]]，假设反向传播的 FLOPs 是前向传播的两倍。

交付物：训练所需小时数，并简要说明理由。

#### 4.4 学习率调度

导致损失最快下降的学习率值在训练过程中经常会变化。在训练 Transformer 时，通常使用学习率调度，即开始时使用较大的学习率，在初期做出更快的更新，然后随着模型训练慢慢衰减到较小的值。在本作业中，我们将实现用于训练 LLaMA [H. Touvron et al., 2023] 的余弦退火调度。

调度器只是一个函数，它接受当前步 \(t\) 和其他相关参数（如初始和最终学习率），并返回用于第 \(t\) 步梯度更新的学习率。最简单的调度是常数函数，它在任何 \(t\) 返回相同学习率。

余弦退火学习率调度接受 (i) 当前迭代 \(t\)，(ii) 最大学习率 \(\alpha_{\mathrm{max}}\)，(iii) 最小（最终）学习率 \(\alpha_{\mathrm{min}}\)，(iv) 预热迭代次数 \(T_{w}\)，以及 (v) 余弦退火的最终迭代 \(T_{c}\)。第 \(t\) 次迭代的学习率定义为：

（预热）如果 \(t< T_{w}\)，则 \(\alpha_{t} = \frac{t}{T_{w}}\alpha_{\mathrm{max}}\)
（余弦退火）如果 \(T_{w}\leq t\leq T_{c}\)，则 \(\alpha_{t} = \alpha_{\mathrm{min}} + \frac{1}{2}\Big(1 + \cos \Big(\frac{t - T_{w}}{T_{c} - T_{w}}\pi \Big)\Big)(\alpha_{\mathrm{max}} - \alpha_{\mathrm{min}})\)
（退火后）如果 \(t > T_{c}\)，则 \(\alpha_{t} = \alpha_{\mathrm{min}}\)。

#### 问题（learning_rate_schedule）：实现带预热的余弦学习率调度（1 分）

---

## 第 34 页

编写一个函数，接受 \(t\)、\(\alpha_{\mathrm{max}}\)、\(\alpha_{\mathrm{min}}\)、\(T_{w}\) 和 \(T_{c}\)，并根据上面定义的调度返回学习率 \(\alpha_{t}\)。然后实现 `[adapters.get_lr_cosine_schedule]` 并确保它通过 `uv run pytest -k test_get_lr_cosine_schedule`。

#### 4.5 梯度裁剪

训练期间，我们有时会遇到产生大梯度的训练样本，这可能使训练不稳定。为缓解此问题，实践中常用的一种技术是梯度裁剪。其思想是在反向传播后、优化器步骤前，对梯度的范数施加限制。

给定梯度（所有参数）\(g\)，我们计算其 \(\ell_{2}\) 范数 \(\| g\|_{2}\)。如果该范数小于最大值 \(M\)，则保持 \(g\) 不变；否则，我们将 \(g\) 按因子 \(\frac{M}{\|g\|_{2} + \epsilon}\) 缩小（其中添加一个小 \(\epsilon\)，如 \(10^{- 6}\)，用于数值稳定性）。注意，结果范数将略小于 \(M\)。

#### 问题（gradient_clipping）：实现梯度裁剪（1 分）

编写一个实现梯度裁剪的函数。你的函数应接受参数列表和最大 \(\ell_{2}\) 范数。它应原地修改每个参数梯度。使用 \(\epsilon = 10^{- 6}\)（PyTorch 默认值）。然后实现适配器 `[adapters.run_gradient_clipping]` 并确保它通过 `uv run pytest -k test_gradient_clipping`。

### 5 训练循环

我们现在终于把目前构建的主要组件放在一起：token 化数据、模型和优化器。

#### 5.1 数据加载器

token 化数据（例如你在 `tokenizer_experiments` 中准备的数据）是单个 token 序列 \(x = (x_{1},\ldots ,x_{n})\)。即使源数据可能由不同文档组成（例如不同网页或源代码文件），常见做法是将它们全部连接成单个 token 序列，并在它们之间添加分隔符（如 `<|endoftext|>` token）。

数据加载器将其转换为批次流，每个批次由 \(B\) 个长度为 \(m\) 的序列组成，并配对相应的下一个 token，长度也为 \(m\)。例如，对于 \(B = 1, m = 3\)，`([x_{2}, x_{3}, x_{4}], [x_{3}, x_{4}, x_{5}])` 可能是一个批次。

以这种方式加载数据简化了训练，原因有几个。首先，任何 \(1 \leq i \leq n - m\) 都给出有效的训练序列，因此采样训练序列很简单。由于所有训练序列长度相同，无需填充输入序列，这提高了硬件利用率（也通过增加批次大小 \(B\)）。最后，我们也不需要加载完整数据集来采样训练数据，使得处理可能无法放入内存的大型数据集变得容易。

#### 问题（data_loading）：实现数据加载（2 分）

交付物：编写一个函数，接受一个 NumPy 数组 \(x\)（整数 token ID 数组）、`batch_size`、`context_length` 和一个 PyTorch 设备字符串（例如 `'cpu'` 或 `'cuda:0'`），并返回一对张量：采样的输入序列和对应的下一个 token 目标。两个

---

## 第 35 页

张量都应具有形状 `(batch_size, context_length)`，包含 token ID，并且都应放在请求的设备上。为了用我们提供的测试测试你的实现，你首先需要实现测试适配器 `[adapters.run_get_batch]`。然后运行 `uv run pytest -k test_get_batch` 测试你的实现。

#### 低资源提示：在 CPU 或 Apple Silicon 上加载数据

如果你计划在 CPU 或 Apple Silicon 上训练 LM，你需要将数据移动到正确的设备（类似地，你之后也应该为模型使用相同设备）。

如果你在 CPU 上，可以使用 `'cpu'` 设备字符串；在 Apple Silicon（M* 芯片）上，可以使用 `'mps'` 设备字符串。

有关 MPS 的更多信息，请查看以下资源：

- `https://docs.pytorch.org/docs/stable/mps.html`
- `https://docs.pytorch.org/docs/stable/notes/mps.html`
- `https://developer.apple.com/documentation/metalperformanceshaders`

如果数据集太大无法加载到内存怎么办？我们可以使用名为 `mmap` 的 Unix 系统调用，它将文件映射到虚拟内存，并在访问该内存位置时惰性加载文件内容。因此，你可以“假装”整个数据集在内存中。NumPy 通过 `np.memmap`（或 `np.load` 的标志 `mmap_mode='r'`，如果你最初用 `np.save` 保存数组）实现这一点，它将返回一个类似 NumPy 数组的对象，在访问条目时按需加载。**训练期间从数据集（即 NumPy 数组）采样时，确保以内存映射模式加载数据集**（通过 `np.memmap` 或 `np.load` 的标志 `mmap_mode='r'`，取决于你保存数组的方式）。还要确保指定与加载数组匹配的 `dtype`。显式验证内存映射数据看起来正确（例如，不包含超出预期词汇表大小的值）可能会有帮助。

#### 5.2 检查点

除了加载数据，我们还需要在训练时保存模型。运行作业时，我们通常希望能够恢复中途停止的训练运行（例如，由于作业超时、机器故障等）。即使一切顺利，我们可能也希望之后访问中间模型（例如，研究训练动态、从不同训练阶段采样模型等）。

检查点应包含恢复训练所需的所有状态。我们当然希望能够至少恢复模型权重。如果使用有状态优化器（如 AdamW），我们还需要保存优化器状态（例如 AdamW 的矩估计）。最后，为了恢复学习率调度，我们需要知道停止时的迭代次数。PyTorch 使保存所有这些变得容易：每个 `nn.Module` 都有一个 `state_dict()` 方法，返回包含所有可学习权重的字典；我们可以用姐妹方法 `load_state_dict()` 恢复这些权重。对于 `torch.optim.Optimizer` 也一样。最后，`torch.save(obj, dest)` 可以将对象（例如包含张量值以及常规 Python 对象如整数的字典）转储到文件（路径）或类文件对象，然后可以用 `torch.load(src)` 加载回内存。

---

## 第 36 页

1

实现以下两个函数来加载和保存检查点：

```python
def save_checkpoint(model, optimizer, iteration, out)
```

应将模型、优化器和迭代次数的所有状态转储到类文件对象 `out` 中。你可以使用模型和优化器的 `state_dict` 方法获取相关状态，并使用 `torch.save(obj, out)` 将 `obj` 转储到 `out`（PyTorch 支持路径或类文件对象）。典型选择是 `obj` 为字典，但你可以使用任何格式，只要之后能加载检查点。

此函数期望以下参数：

- `model: torch.nn.Module`
- `optimizer: torch.optim.Optimizer`
- `iteration: int`
- `out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`

```python
def load_checkpoint(src, model, optimizer)
```

应从 `src`（路径或类文件对象）加载检查点，然后从该检查点恢复模型和优化器状态。你的函数应返回保存到检查点中的迭代次数。你可以使用 `torch.load(src)` 恢复你在 `save_checkpoint` 实现中保存的内容，并使用模型和优化器中的 `load_state_dict` 方法将它们返回到之前的状态。

此函数期望以下参数：

- `src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`
- `model: torch.nn.Module`
- `optimizer: torch.optim.Optimizer`

实现 `[adapters.run_save_checkpoint]` 和 `[adapters.run_load_checkpoint]` 适配器，并确保它们通过 `uv run pytest -k test_checkpointing`。

#### 5.3 训练循环

现在，终于到了把你实现的所有组件放在一起，放入主训练脚本的时候了。让以不同超参数启动训练运行变得容易（例如通过命令行参数）会很有回报，因为你之后会多次这样做，以研究不同选择如何影响训练。

#### 问题（training together）：把它们放在一起（4 分）

交付物：编写一个脚本，运行训练循环，在用户提供的输入上训练模型。特别地，我们建议你的训练脚本至少允许以下内容：

- 能够配置和控制各种模型和优化器超参数。
- 使用 `np.memmap` 内存高效地加载大型训练和验证数据集。
- 将检查点序列化到用户提供的路径。

---

## 第 37 页

- 定期记录训练和验证性能（例如到控制台和/或 Weights and Biases 等外部服务）。\(^{9}\)

### 6 生成文本

现在我们可以训练模型了，最后需要的部分是从模型生成文本的能力。回想一下，语言模型接受（可能批量的）长度为 `sequence_length` 的整数序列，并产生大小为 `(sequence_length, vocab_size)` 的矩阵，其中序列的每个元素是预测该位置之后下一个 token 的概率分布。我们现在将编写一些函数，将其转化为新序列的采样方案。

#### Softmax

按标准惯例，语言模型输出是最终线性层的输出（“logits”），所以我们必须通过 softmax 操作将其转换为归一化概率，我们之前在第 10 节方程中见过。

#### 解码

为了从模型生成文本（解码），我们将向模型提供前缀 token 序列（“提示”），并要求它产生词汇表上的概率分布，预测序列中的下一个 token。然后，我们将从词汇项上的该分布中采样，以确定下一个输出 token。

具体来说，解码过程的一步应接受序列 \(x_{1\ldots t}\) 并通过以下方程返回 token \(x_{t + 1}\)：

\[
\begin{array}{r}
P(x_{t + 1} = i\mid x_{1\ldots t}) = \frac{\exp(v_i)}{\sum_j\exp(v_j)} \\
v = \mathrm{TransformerLM}(x_{1\ldots t})_t\in \mathbb{R}^{\mathrm{vocab\_size}}
\end{array} \quad (21)
\]

其中 `TransformerLM` 是我们的模型，它接受长度为 `sequence_length` 的序列，并产生大小为 `(sequence_length, vocab_size)` 的矩阵，我们取该矩阵的最后一个元素，因为我们在寻找第 \(t\) 个位置的下一个 token 预测。

这给了我们一个基本解码器：通过重复从这些一步条件分布中采样（将之前生成的输出 token 追加到下一个解码时间步的输入中），直到生成序列结束 token `<|endoftext|>`（或用户指定的最大生成 token 数）。

#### 解码器技巧

我们将用小型模型做实验，小型模型有时会生成质量很低的文本。两个简单的解码器技巧可以帮助解决这些问题。首先，在温度缩放中，我们用温度参数 \(\tau\) 修改 softmax，新的 softmax 是

\[
\mathrm{softmax}(v,\tau)_i = \frac{\exp(v_i / \tau)}{\sum_{j = 1}^{\mathrm{vocab\_size}}\exp(v_j / \tau)}. \quad (23)
\]

注意，设置 \(\tau \to 0\) 会使 \(v\) 的最大元素占主导，softmax 输出变成集中在该最大元素上的一热向量。

---

## 第 38 页

第二，另一个技巧是核采样或 top-p 采样，我们通过截断低概率 token 来修改采样分布。令 \(q\) 为从大小为 `vocab_size` 的（温度缩放）softmax 得到的概率分布。使用超参数 \(p\) 的核采样根据以下方程产生下一个 token：

\[
P(x_{t + 1} = i\mid q) = \left\{ \begin{array}{ll}\frac{q_i}{\sum_{j\in V(p)}q_j} & \mathrm{if} i\in V(p)\\ 0 & \mathrm{otherwise} \end{array} \right. \quad (24)
\]

其中 \(V(p)\) 是最小的索引集合，使得 \(\sum_{j\in V(p)}q_j\geq p\)。你可以通过首先按大小排序概率分布 \(q\)，并选择最大的词汇元素直到达到目标水平 \(p\)，轻松计算该量。

#### 问题（decoding）：解码（3 分）

交付物：实现一个从语言模型解码的函数。我们建议你支持以下功能：

- 为用户提供的提示生成补全（即接受一些 \(x_{1\dots t}\) 并采样补全，直到遇到 `<|endoftext|>` token）。
- 允许用户控制最大生成 token 数。
- 给定所需温度值，在采样前对预测的下一个 token 分布应用 softmax 温度缩放。
- Top-\(p\) 采样（A. Holtzman et al., 2020，也称为核采样），给定用户指定的阈值。

### 7 实验

现在是时候把所有东西放在一起，在预训练数据集上训练（小型）语言模型了。

#### 7.1 如何运行实验和交付物

理解 Transformer 架构组件背后原理的最佳方式是实际修改它并自己运行。没有替代动手经验的东西。

为此，能够快速、一致地实验并记录你做了什么很重要。为了快速实验，我们将在小规模模型（约 17M 总参数）和简单数据集（TinyStories）上运行许多实验。为了保持一致，你将以系统的方式消融组件并改变超参数；为了记录，我们将要求你提交实验日志和每个实验相关的学习曲线。

为了能够提交损失曲线，确保定期评估验证损失，并记录步数和墙钟时间。你可能会发现 Weights and Biases 等日志基础设施有帮助。

#### 问题（experiment_log）：实验日志（3 分）

对于你的训练和评估代码，创建实验跟踪基础设施，允许你跟踪实验和损失曲线，相对于梯度步数和墙钟时间。

---

## 第 39 页

1

#### 7.2 TinyStories

我们将从非常简单的数据集（TinyStories；R. Eldan et al. [1]）开始，模型将快速训练，我们可以看到一些有趣的行为。获取此数据集的说明见第 1 节。该数据集的样子示例如下。

#### 示例（tinystories_example）：TinyStories 中的一个示例

从前，有一个叫 Ben 的小男孩。Ben 喜欢探索周围的世界。他看到了许多奇妙的东西，比如商店里陈列的美丽花瓶。一天，Ben 走过商店时，看到一个非常特别的花瓶。当 Ben 看到它时，他惊叹不已！他说：“哇，那真是一个很棒的花瓶！我能买它吗？”店主笑着说：“当然可以。你可以把它带回家，向你所有的朋友展示它有多棒！”于是 Ben 把花瓶带回家，他为它感到非常自豪！他叫朋友们过来，向他们展示这个神奇的花瓶。他所有的朋友都认为花瓶很漂亮，不敢相信 Ben 有多幸运。这就是 Ben 如何在商店里找到一个神奇花瓶的故事！

#### 7.2.1 超参数调优

我们会告诉你一些非常基本的起始超参数，并要求你找到其他一些效果好的设置。

- 词汇表大小 10000。典型词汇表大小在数万到数十万。你应该改变它，看看词汇表和模型行为如何变化。
- 上下文长度 256。像 TinyStories 这样的简单数据集可能不需要长序列长度，但后面的 OpenWebText 数据可能想要改变。尝试改变它，看看对每次迭代运行时间和最终困惑度的影响。
- `d_model` 512。这比许多小型 Transformer 论文中使用的 768 维略小，但这会使事情更快。
- `d_ff` 1344。这大约是 \(\frac{8}{3} d_{\mathrm{model}}\)，同时是 64 的倍数，对 GPU 性能有好处。
- RoPE theta 参数 \(\Theta\) 10000。
- 层数和头数 4 层，16 头。一起，这将给出约 17M 非嵌入参数，是一个相当小的 Transformer。
- 处理的总 token 数 327,680,000（你的批次大小 × 总步数 × 上下文长度应大约等于该值）。

你应该进行一些试错，为以下其他超参数找到好的默认值：学习率、学习率预热、其他 AdamW 超参数 \((\beta_{1},\beta_{2},\epsilon)\) 和权重衰减。你可以在 D. P. Kingma et al. [22] 中找到此类超参数的一些典型选择。

#### 7.2.2 把它们放在一起

现在你可以通过获取训练好的 BPE 分词器、token 化训练数据集，并在你写的训练循环中运行，把所有东西放在一起。重要提示：如果你的实现正确且高效，上述超参数应该导致在 1 台 B200 上大约 20-30 分钟的运行时间。

---

## 第 40 页

7.2.3 调试模型架构的技巧和诀窍

我们强烈建议你熟悉 IDE 内置调试器（例如 VSCode/Zed），与用 print 语句调试相比，这会节省时间。如果你使用文本编辑器，可以使用 `ipdb` 之类的东西。调试模型架构时，其他一些好做法是：

开发任何神经网络架构时，常见的第一个步骤是对单个小批次过拟合。如果你的实现正确，你应该能够快速将训练损失降到接近零。在模型各组件中设置调试断点，检查中间张量的形状，确保它们符合你的预期。监控激活、模型权重和梯度的范数，确保它们没有爆炸或消失。

#### 问题（learning_rate）：调整学习率（2 B200 小时）（3 分）

学习率是最重要的调优超参数之一。基于你训练的基础模型，回答以下问题：

(a) 对学习率进行超参数扫描，并报告最终损失（或如果优化器发散则注明发散）。

交付物：多个学习率相关的学习曲线。解释你的超参数搜索策略。

交付物：一个在 TinyStories 上验证损失（每 token）至多 1.45 的模型。

#### 低资源提示：在 CPU 或 Apple Silicon 上训练几步

如果你在 CPU 或 MPS 上运行，你应该改为将处理的总 token 数减少到 40,000,000，这足以产生相当流畅的文本。你也可以将目标验证损失从 1.45 提高到 2.00。

在 M4 Max 芯片和 36 GB RAM 上运行我们的解决方案代码并调优学习率，我们使用批次大小 × 总步数 × 上下文长度 = \(32\times 5000\times 256 = 40,960,000\) tokens，这在 CPU 上需要 1 小时 22 分钟，在 MPS 上需要 36 分钟。在第 5000 步，我们达到验证损失 1.80。

一些额外提示：

- 使用 \(N\) 个训练步时，我们建议调整余弦学习率衰减调度，使其衰减（即达到最小学习率）恰好在第 \(N\) 步终止。
- 使用 MPS 时，不要使用 TF32 内核，即不要设置 `torch.set_float32_matmul_precision('high')`，就像你在 CUDA 设备上可能会做的那样。我们尝试在 MPS 上启用 TF32 内核（torch 版本 2.9.0），发现后端有时会使用静默损坏的内核，导致训练不稳定。

---

## 第 41 页

- 你可以通过用 `torch.compile` JIT 编译模型来加速训练。具体来说：
  - 在 CPU 上，用 `model = torch.compile(model)` 编译模型。
  - 在 MPS 上，你可以用 `model = torch.compile(model, backend="aot_eager")` 在一定程度上优化反向传播。截至 torch 版本 2.9.0，MPS 不支持 Inductor 编译。

(b) 民间智慧是，最佳学习率“处于稳定边缘”。研究学习率发散点与你的最佳学习率之间的关系。

交付物：增加学习率的学习曲线，至少包括一次发散运行，并分析这与收敛率的关系。

现在让我们改变批次大小，看看训练会发生什么。批次大小很重要——它们让我们通过更大的矩阵乘法从 GPU 获得更高效率，但总是想要大批次大小是真的吗？让我们运行一些实验来找出答案。

#### 问题（batch_size_experiment）：批次大小变化（1 B200 小时）（1 分）

将批次大小从 1 变化到 GPU 内存限制。尝试至少几个中间大小，包括典型大小如 64 和 128。

交付物：不同批次大小运行的学习曲线。如有必要，学习率应再次优化。

交付物：几句话讨论你对批次大小及其对训练影响的发现。

有了解码器，我们现在可以生成文本！我们将从模型生成并看看它有多好。作为参考，你应该得到至少和下面示例一样好的输出。

#### 示例（ts_generate_example）：TinyStories 语言模型的示例输出

从前，有一个叫 Lily 的漂亮女孩。她喜欢吃口香糖，尤其是大黑的那种。一天，Lily 的妈妈让她帮忙做晚饭。Lily 非常兴奋！她喜欢帮妈妈。Lily 的妈妈做了一大锅汤当晚餐。Lily 非常高兴地说：“谢谢你，妈妈！我爱你。”她帮妈妈把汤倒进一个大碗里。晚饭后，Lily 的妈妈又做了一些美味的汤。Lily 很喜欢！她说：“谢谢你，妈妈！这汤真好吃！”她妈妈笑着说：“我很高兴你喜欢，Lily。”他们做完饭，继续一起做饭。故事结束。

#### 低资源提示：在 CPU 或 Apple Silicon 上生成文本

---

## 第 42 页

如果你改为使用处理 40M token 的低资源配置，你应该看到生成内容仍然像英语，但不如上面流畅。例如，我们训练 40M token 的 TinyStories 语言模型的示例输出如下：

从前，有一个叫 Sue 的小女孩。Sue 有一颗她非常喜欢的牙齿。那是他最好的头。一天，Sue 出去散步，遇到了一只瓢虫！他们成了好朋友，一起在路上玩。

“嘿，Polly！我们出去吧！”Tim 说。Sue 看着天空，发现很难找到一种跳舞发光的方式。她微笑着同意帮助会说话的！

当 Sue 看着天空移动，那是什么。她

以下是确切的问题陈述和我们要求的内容：

#### 问题（generate）：生成文本（1 分）

使用你的解码器和训练好的检查点，报告你的模型生成的文本。你可能需要调整解码器参数（温度、top-p 等）以获得流畅输出。

交付物：至少 256 个 token 的文本转储（或直到第一个 `<|endoftext|>` token），并简要评论该输出的流畅性以及至少两个影响输出好坏的因素。

#### 7.3 消融和架构修改

理解 Transformer 的最佳方式是实际修改它并看看它的行为。我们现在将做一些简单的消融和修改。

#### 消融 1：层归一化

人们常说层归一化对 Transformer 训练的稳定性很重要。但也许我们想冒险。让我们从每个 Transformer 块中移除 RMSNorm，看看会发生什么。

#### 问题（layer_norm_ablation）：移除 RMSNorm 并训练（0.5 B200 小时）（1 分）

从 Transformer 中移除所有 RMSNorm 并训练。在之前的最佳学习率下会发生什么？你能通过使用更低的学习率获得稳定性吗？

交付物：移除 RMSNorm 并训练时的学习曲线，以及最佳学习率的学习曲线。

交付物：几句话评论 RMSNorm 的影响。

现在让我们研究另一个乍看任意的层归一化选择。Pre-norm Transformer 块定义为

\[
\begin{array}{rl}
z = x + \mathrm{MultiHeadSelfAttention}(\mathrm{RMSNorm}(x))\\
y = z + \mathrm{FFN}(\mathrm{RMSNorm}(z)).
\end{array} \quad (25)
\]

---

## 第 43 页

这是对原始 Transformer 架构的少数共识修改之一，原始 Transformer 使用 post-norm 方法：

\[
\begin{array}{rl}
z = \mathrm{RMSNorm}(x + \mathrm{MultiHeadSelfAttention}(x))\\
y = \mathrm{RMSNorm}(z + \mathrm{FFN}(z)).
\end{array} \quad (27)
\]

让我们回到 post-norm 方法，看看会发生什么。

#### 问题（pre_norm_ablation）：实现 post-norm 并训练（0.5 B200 小时）（1 分）

将你的 pre-norm Transformer 实现修改为 post-norm。用 post-norm 模型训练，看看会发生什么。

交付物：post-norm Transformer 与 pre-norm 相比的学习曲线。

我们看到层归一化对 Transformer 行为有重大影响，甚至层归一化的位置也很重要。

#### 消融 2：位置嵌入

接下来我们将研究位置嵌入对模型性能的影响。具体来说，我们将比较我们的基础模型（带 RoPE）与完全不包含位置嵌入（NoPE）。事实证明，decoder-only Transformer，即具有因果掩码的 Transformer，理论上可以在没有显式提供位置嵌入的情况下推断相对或绝对位置信息 [Y.-H. H. Tsai et al., 2019; A. Kazemnejad et al., 2023]。我们现在将实证测试 NoPE 与 RoPE 的表现。

#### 问题（no_pos_emb）：实现 NoPE（0.5 B200 小时）（1 分）

修改你的带 RoPE 的 Transformer 实现，完全移除位置嵌入信息，看看会发生什么。

交付物：比较 RoPE 和 NoPE 性能的学习曲线。

#### 消融 3：SwiGLU vs. SiLU

接下来，我们将遵循 [N. Shazeer [20] 并测试门控在前馈网络中的重要性，通过比较 SwiGLU 前馈网络与使用 SiLU 激活但没有门控线性单元（GLU）的前馈网络：

\[
\mathrm{FFN}_{\mathrm{SiLU}}(x) = W_2\mathrm{SiLU}(W_1x). \quad (29)
\]

回想一下，在我们的 SwiGLU 实现中，我们将内部前馈层的维度设置为大约 \(d_{\mathrm{ff}} = \frac{8}{3} d_{\mathrm{model}}\)（同时确保 \(d_{\mathrm{ff}}\mathrm{mod}64 = 0\)，以利用 GPU 张量核心）。在这个消融基线中，你的 \(\mathrm{FFN}_{\mathrm{SiLU}}\) 实现应改为设置 \(d_{\mathrm{ff}} = 4\times d_{\mathrm{model}}\)，以大约匹配默认 SwiGLU 前馈网络的参数数量（后者有三个而不是两个权重矩阵）。

#### 问题（swiGLU_ablation）：SwiGLU vs. SiLU（0.5 B200 小时）（1 分）

---

## 第 44 页

1 交付物：比较 SwiGLU 和 SiLU 前馈网络性能的学习曲线，参数数量大致匹配。交付物：几句话讨论你的发现。

#### 低资源提示：资源有限的在线学生应在 TinyStories 上测试修改

在作业剩余部分，我们将转向更大规模、更嘈杂的网络数据集（OpenWebText），实验架构修改并（可选）向课程排行榜提交。

在 OpenWebText 上训练 LM 达到流畅需要很长时间，所以我们建议 GPU 访问有限的在线学生继续在 TinyStories 上测试修改（使用验证损失作为评估性能的指标）。

#### 7.4 在 OpenWebText 上运行

我们现在转向从网络爬取创建的一个更标准的预训练数据集。OpenWebText [A. Gokaslan et al., 2019] 的一个小样本也作为单个文本文件提供：访问此文件的说明见 [第 1 节]。

以下是 OpenWebText 的一个示例。注意文本更加真实、复杂和多样。你可能想浏览训练数据集，了解网络爬取语料的训练数据是什么样子。

#### 示例（owt_example）：一个 OWT 示例

Baseball Prospectus 技术总监 Harry Pavlidis 在雇用 Jonathan Judge 时冒了风险。

Pavlidis 知道，正如 Alan Schwarz 在《The Numbers Game》中写道，“美国文化中没有哪个角落比棒球运动员的表现被更精确地计数、更热情地量化。”点击几下，你就能发现 Noah Syndergaard 的快球在到达本垒板途中每分钟旋转超过 2,100 次，Nelson Cruz 在 2016 年合格击球手中拥有比赛最高的平均出口速度，以及无数其他似乎从电子游戏或科幻小说中撕下来的小信息。数据不断上涨的海洋赋予棒球文化中一个越来越重要的角色：分析爱好者。

这种赋权伴随着更多审视——不仅对测量，也对背后的人和出版物。对于 Baseball Prospectus，Pavlidis 非常了解伴随定量不完美而来的反弹。他也知道网站的接球指标需要重做，并且需要一个有学问的头脑——一个能解决复杂统计建模问题的人——来完成这项工作。

“他把我们吓坏了。” Harry Pavlidis

Pavlidis 根据 Judge 的写作以及他们在网站赞助的球场活动上的互动，有一种预感，Judge“懂了”。[...]

---

## 第 45 页

注意：你可能需要为此实验重新调整超参数，例如学习率或批次大小。

#### 问题（main_experiment）：在 OWT 上实验（2 B200 小时）（2 分）

在 OpenWebText 上训练你的语言模型，使用与 TinyStories 相同的模型架构和总训练迭代次数。这个模型表现如何？

交付物：你的语言模型在 OpenWebText 上的学习曲线。描述与 TinyStories 损失的差异——我们应该如何解释这些损失？

交付物：来自 OpenWebText LM 的生成文本，格式与 TinyStories 输出相同。这段文本的流畅性如何？为什么即使模型和计算预算与 TinyStories 相同，输出质量却更差？

#### 7.5 你自己的修改 + 排行榜

恭喜你走到这一步。你快完成了！你现在将尝试改进 Transformer 架构，并看看你的超参数和架构如何在班级中与其他学生相比。

#### 排行榜规则

除了以下内容，没有其他限制：

- 运行时间：你的提交在 B200 上最多可运行 45 分钟。如果你使用 SLURM 或 Modal，可能想在提交脚本中强制执行这一点。
- 数据：你只能使用我们提供的 OpenWebText 训练数据集。

否则，你可以随心所欲。

如果你在寻找一些实现想法，可以查看以下资源：

- 最先进的开源 LLM 系列，如 Llama 3 [A. Grattafiori et al., 2024] 或 Qwen 2.5 [A. Yang et al., 2024]。
- NanoGPT speedrun 仓库（`github.com/KellerJordan/modded-nanogpt`），社区成员发布了许多有趣的修改，用于“速通”小规模语言模型预训练。例如，一个可以追溯到原始 Transformer 论文的常见修改是将输入和输出嵌入的权重绑定在一起（见 [A. Vaswani et al. [8]（第 3.4 节）和 [A. Chowdhery et al. [16]（第 2 节））。如果你尝试权重绑定，可能需要降低嵌入/LM head 初始化的标准差。

在尝试完整 45 分钟运行之前，你可能想在 OpenWebText 的小子集或 TinyStories 上测试这些修改。

需要说明的是，我们确实注意到，你在排行榜中发现有效的一些修改可能不能泛化到更大规模预训练。我们将在课程的缩放定律单元中进一步探讨这个想法。

#### 问题（leaderboard）：排行榜（10 B200 小时）（6 分）

---

## 第 46 页

1

你将在上述排行榜规则下训练一个模型，目标是在 0.75 B200 小时内最小化语言模型的验证损失。

交付物：记录的最终验证损失、相关学习曲线（清晰显示墙钟时间 x 轴小于 45 分钟），以及你做了什么的描述。我们期望排行榜提交至少超过 5.0 损失的朴素基线。在此提交到排行榜：`github.com/stanford-cs336/assignment1-basics-leaderboard`。

### 参考文献

[1] R. Eldan and Y. Li, “TinyStories: How Small Can Language Models Be and Still Speak Coherent English?” 2023.
[2] A. Gokaslan, V. Cohen, E. Pavlick, and S. Tellex, “OpenWebText corpus.” 2019.
[3] R. Sennrich, B. Haddow, and A. Birch, “Neural Machine Translation of Rare Words with Subword Units,” in Proc. of ACL, 2016.
[4] C. Wang, K. Cho, and J. Gu, “Neural Machine Translation with Byte-Level Subwords.” 2019.
[5] P. Gage, “A new algorithm for data compression,” C Users Journal, vol. 12, no. 2, pp. 23-38, Feb. 1994.
[6] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, “Language Models are Unsupervised Multitask Learners.” 2019.
[7] A. Radford, K. Narasimhan, T. Salimans, and I. Sutskever, “Improving Language Understanding by Generative Pre-Training.” 2018.
[8] A. Vaswani et al., “Attention is All you Need,” in Proc. of NeurIPS, 2017.
[9] T. Q. Nguyen and J. Salazar, “Transformers without Tears: Improving the Normalization of Self-Attention,” in Proc. of IWSWLT, 2019.
[10] R. Xiong et al., “On Layer Normalization in the Transformer Architecture,” in Proc. of ICML, 2020.
[11] J. L. Ba, J. R. Kiros, and G. E. Hinton, “Layer Normalization.” 2016.
[12] H. Touvron et al., “LLaMA: Open and Efficient Foundation Language Models.” 2023.
[13] B. Zhang and R. Sennrich, “Root Mean Square Layer Normalization,” in Proc. of NeurIPS, 2019.
[14] A. Grattafiori et al., “The Llama 3 Herd of Models.” [Online]. Available: `https://arxiv.org/abs/2407.21783`
[15] A. Yang et al., “Qwen2.5 Technical Report,” arXiv preprint arXiv:2412.15115, 2024.
[16] A. Chowdhery et al., “PaLM: Scaling Language Modeling with Pathways.” 2022.
[17] D. Hendrycks and K. Gimpel, “Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units.” 2016.
[18] S. Elfwing, E. Uchibe, and K. Doya, “Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning.” [Online]. Available: `https://arxiv.org/abs/1702.03118`
[19] Y. N. Dauphin, A. Fan, M. Auli, and D. Grangier, “Language Modeling with Gated Convolutional Networks.” [Online]. Available: `https://arxiv.org/abs/1612.08083`
[20] N. Shazeer, “GLU Variants Improve Transformer.” 2020.

---

## 第 47 页

[21] J. Su, Y. Lu, S. Pan, B. Wen, and Y. Liu, “RoFormer: Enhanced Transformer with Rotary Position Embedding,” 2021.
[22] D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” in Proc. of ICLR, 2015.
[23] I. Loshchilov and F. Hutter, “Decoupled Weight Decay Regularization,” in Proc. of ICLR, 2019.
[24] T. B. Brown et al., “Language Models are Few-Shot Learners,” in Proc. of NeurIPS, 2020.
[25] J. Kaplan et al., “Scaling Laws for Neural Language Models,” 2020.
[26] J. Hoffmann et al., “Training Compute-Optimal Large Language Models,” 2022.
[27] A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi, “The Curious Case of Neural Text Degeneration,” in Proc. of ICLR, 2020.
[28] Y.-H. H. Tsai, S. Bai, M. Yamada, L.-P. Morency, and R. Salakhutdinov, “Transformer Dissection: An Unified Understanding for Transformer's Attention via the Lens of Kernel,” in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), K. Inui, J. Jiang, V. Ng, and X. Wan, Eds., Hong Kong, China: Association for Computational Linguistics, Nov. 2019, pp. 4344-4353. doi: 10.18653/v1/D19-1443.
[29] A. Kazemnejad, I. Padhi, K. Natesan, P. Das, and S. Reddy, “The Impact of Positional Encoding on Length Generalization in Transformers,” in Thirty-seventh Conference on Neural Information Processing Systems, 2023. [Online]. Available: `https://openreview.net/forum?id=Dnr12gcjz1`

（全文完）