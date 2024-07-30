import os

import numpy as np
import torch
from torch.utils.data import Dataset


class VAE_Dataset(Dataset):
    def __init__(self, dir):
        names = os.listdir(dir)
        self.files = [os.path.join(dir, n) for n in names]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        obs = np.load(self.files[index])['obs']
        obs = torch.tensor(obs, dtype=torch.float).permute(0, 3, 1, 2)

        return obs