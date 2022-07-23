from torch.utils.data import DataLoader

def construct_dataloader(dataset, **kwargs):
    dataloader = DataLoader(dataset, shuffle=True, pin_memory=True, **kwargs)
    return dataloader