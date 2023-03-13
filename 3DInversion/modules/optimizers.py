import torch

def get_optimizer(optimizer_str):
    if optimizer_str == "Adam":
        optimizer = torch.optim.Adam

    return optimizer