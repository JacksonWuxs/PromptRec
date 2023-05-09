

import numpy as np
import torch as tc


from .core.dataset import BaseData


class CombineData(tc.utils.data.Dataset):
    def __init__(self, datas):
        self.datas = []
        for data in datas:
            assert isinstance(data, BaseData)
            self.datas.append(data)

    def __iter__(self):
        for data in self.datas:
            for row in data:
                yield row

    def __len__(self):
        return sum(map(len, self.datas))

    def __getitem__(self, idx):
        for data in self.datas:
            if len(data) > idx:
                break
            idx -= len(data)
        return data[idx]

    def get_max_fid(self):
        return max(data.get_max_fid() for data in self.datas)
        
