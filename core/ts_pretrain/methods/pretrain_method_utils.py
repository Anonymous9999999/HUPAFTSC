import torch
import hashlib
import numpy as np
from torch.utils.data import Dataset

class TsGeneralPretrainDataset(Dataset):
    def __init__(self,
                 sktime_data: np.ndarray
                 ):
        self.sktime_data = sktime_data

    @property
    def hash_value(self):
        return hashlib.md5(str(self.sktime_data).encode('utf-8')).hexdigest()

    def __getitem__(self, ind):
        X = self.sktime_data[ind]
        return torch.from_numpy(X)

    def __len__(self):
        return len(self.sktime_data)