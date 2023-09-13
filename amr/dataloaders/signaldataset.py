import torch.utils.data as data
import torch
import numpy as np
from .transform import *
__all__ = ["SignalDataset"]


class SignalDataset(data.Dataset):
    def __init__(self, X, mods, snrs, datapreprocess, ismulti):
        super(SignalDataset, self).__init__()
        self.X = X
        self.mods = mods
        self.snrs = snrs
        self.datapreprocess = datapreprocess
        self.ismulti = ismulti

    def __getitem__(self, index):
        return torch.tensor(self.datapreprocess(self.X[index])), self.mods[index], self.snrs[index]

    def __len__(self):
        return self.X.shape[0]
