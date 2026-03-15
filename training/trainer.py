import torch
#from tqdm import tqdm


class Trainer:

    def __init__(
        self,
        model,
        dataloader,
        optimizer,
        loss_fn,
        device="cpu"
    ):

        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

        self.model.to(device)

    def train_epoch(self):

        self.model.train()

        total_loss = 0

        for batch in self.dataloader:

            batch = batch.to(self.device)

            # forward
            logits = self.model(batch)

            # loss
            loss = self.loss_fn(logits, batch)

            # backward
            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.dataloader)

        return avg_loss

    def train(self, epochs):

        for epoch in range(epochs):

            loss = self.train_epoch()

            print(f"Epoch {epoch+1} Loss {loss:.4f}")