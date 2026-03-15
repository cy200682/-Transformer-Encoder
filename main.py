import torch

from text_tokenizer.tokenizer import Tokenizer
from dataset.dataset import TextDataset
from dataset.dataloader import build_dataloader

from model.transformer_model import TransformerModel

from training.loss import TransformerLoss
from training.optimizer import build_optimizer
from training.trainer import Trainer


def main():

    # ---------------------
    # 1 准备数据
    # ---------------------

    texts = [
        "I love deep learning",
        "Transformer is powerful",
        "Natural language processing is interesting",
        "I enjoy studying AI",
        "PyTorch makes deep learning easy"
    ]

    # ---------------------
    # 2 构建 text_tokenizer
    # ---------------------

    tokenizer = Tokenizer(max_vocab_size=10000)

    tokenizer.build_vocab(texts)

    word2idx = tokenizer.word2idx

    vocab_size = len(word2idx)

    print("vocab size:", vocab_size)

    # ---------------------
    # 3 构建 dataset
    # ---------------------

    dataset = TextDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_len=10
    )

    # ---------------------
    # 4 dataloader
    # ---------------------

    dataloader = build_dataloader(
        dataset,
        batch_size=2
    )

    # ---------------------
    # 5 构建模型
    # ---------------------

    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=8,
        num_layers=4,
        d_ff=512,
        max_len=10
    )

    # ---------------------
    # 6 loss
    # ---------------------

    loss_fn = TransformerLoss()

    # ---------------------
    # 7 optimizer
    # ---------------------

    optimizer = build_optimizer(model)

    # ---------------------
    # 8 trainer
    # ---------------------

    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device="cpu"
    )

    # ---------------------
    # 9 开始训练
    # ---------------------

    trainer.train(epochs=10)


if __name__ == "__main__":
    main()