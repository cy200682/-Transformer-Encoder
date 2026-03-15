import torch


def build_optimizer(model, lr=1e-4):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    return optimizer