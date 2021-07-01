from torch.utils.data import TensorDataset
import numpy as np
import torch

class Velocity(TensorDataset):
    def __init__(self, path, transform=None):
        self.tensors = np.load(path)

        mini = np.min(self.tensors)
        maxi = np.max(self.tensors)
        self.tensors = (self.tensors - mini) / (maxi - mini)

        self.tensors = torch.from_numpy(self.tensors).unsqueeze(1)

        self.transform = transform

    def __len__(self):
        return self.tensors.size(0)

    def __getitem__(self, index):
        sample = self.tensors[index]

        if self.transform:
            sample = self.transform(sample)

        target = 0 #dummy target

        return sample, target