from torch.utils.data import DataLoader

def build_dataloader(dataset, batch_size=32, shuffle=True):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return dataloader