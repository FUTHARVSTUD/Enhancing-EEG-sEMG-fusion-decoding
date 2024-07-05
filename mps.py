import torch

if torch.backends.mps.is_built():
    print("MPS is built")