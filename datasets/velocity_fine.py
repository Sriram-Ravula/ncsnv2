from torch.utils.data import TensorDataset
import numpy as np
import torch

class Velocity(TensorDataset):
    def __init__(self, path, transform=None):
        self.tensors = torch.load(path)
        self.transform = transform

    def __len__(self):
        return self.tensors.size(0)

    def __getitem__(self, index):
        sample = self.tensors[index]

        if self.transform:
            sample = self.transform(sample)

        return sample, index
        