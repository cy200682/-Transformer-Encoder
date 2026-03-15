# Transformer Encoder 从零实现（PyTorch）

本项目使用 **PyTorch 从零实现 Transformer Encoder 架构**，包含完整的 NLP 数据处理与训练流程，用于加深对 Attention 机制和 Transformer 模型结构的理解。

项目没有依赖 HuggingFace 等高层框架，核心模块均为手动实现。

---

# 项目特点

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
