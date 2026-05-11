# Transformer Encoder 从零实现（PyTorch）

本项目使用 **PyTorch 从零实现 Transformer Encoder 架构**，包含完整的 NLP 数据处理与训练流程，用于加深对 Attention 机制和 Transformer 模型结构的理解。

项目没有依赖 HuggingFace 等高层框架，核心模块均为手动实现。

---

# 项目特点

我实现了一个基于 PyTorch 的 Mini Transformer 项目，核心是一个 Encoder-only 的 Transformer 模型，用于进行基础的语言建模任务。

整个项目从底层开始实现，没有直接调用现成 Transformer API，而是自己实现了 Tokenizer、Embedding、Positional Encoding、Scaled Dot-Product Attention、Multi-Head Attention、Feed Forward Network 以及多层 Transformer Encoder。

在数据处理部分，我实现了文本清洗、词表构建以及 token 到 index 的映射，并通过 Dataset 和 DataLoader 完成批量数据加载。

模型部分主要基于 Self-Attention 机制，通过多头注意力建模 token 之间的上下文关系，并结合残差连接、LayerNorm 和 FeedForward 网络提升训练稳定性和表达能力。

训练阶段使用 CrossEntropyLoss 作为 token-level 分类损失，并使用 Adam 优化器进行训练，同时加入 Dropout 防止过拟合。

在项目过程中，我比较深入地理解了 Transformer 的核心机制，比如 attention 的计算过程、为什么 Transformer 能并行化、位置编码的作用，以及 padding mask 对 attention 的影响等。目前项目还存在一些可以继续优化的地方，例如完善 padding mask、加入 causal mask，并扩展 Decoder 结构以支持 GPT-style 的文本生成。

- 从零实现 Transformer Encoder 架构
- 实现完整 NLP pipeline（Tokenizer → Dataset → DataLoader）
- 实现 Attention 机制核心模块
- 代码结构模块化，便于扩展与实验

---

# Transformer 结构

项目实现的核心组件包括：

- Token Embedding
- Positional Encoding（正弦位置编码）
- Scaled Dot-Product Attention
- Multi-Head Attention
- Position-wise Feed Forward Network
- Transformer Encoder Layer
- Transformer Encoder Stack

整体结构：


Input Text
↓
Tokenizer
↓
Token Embedding
↓
Positional Encoding
↓
Transformer Encoder (Multi-layer)
↓
Linear Output Layer


---

# 项目结构


transformer_sentiment/

tokenizer/
tokenizer.py # 文本分词与词表构建

dataset/
dataset.py # 自定义 Dataset
dataloader.py # DataLoader 构建

model/
embedding.py
positional_encoding.py
attention.py
multihead_attention.py
feed_forward.py
encoder_layer.py
transformer_encoder.py
transformer_model.py

training/
loss.py
optimizer.py
trainer.py

main.py # 训练入口


---

# 运行方式

安装依赖：


pip install torch tqdm


运行训练：


python main.py


训练示例输出：


Epoch 1 Loss 3.2461
Epoch 2 Loss 2.9782
Epoch 3 Loss 2.7675
Epoch 4 Loss 2.5748
Epoch 5 Loss 2.3965
Epoch 6 Loss 2.2442
Epoch 7 Loss 2.0945
Epoch 8 Loss 2.0458
Epoch 9 Loss 1.8312
Epoch 10 Loss 1.7615


---

# 技术栈

- Python
- PyTorch
- Deep Learning
- Natural Language Processing
- Transformer

---

# 项目目的

本项目旨在通过 **从零实现 Transformer Encoder**，深入理解以下核心概念：

- Self-Attention 机制
- Multi-Head Attention
- Positional Encoding
- Transformer Encoder 架构
- NLP 数据处理流程
