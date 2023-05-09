
import numpy as np
import torch as tc

class RandomRecommend(tc.nn.Module):
    def __init__(self, device, *args, **kwrds):
        tc.nn.Module.__init__(self)
        self.device = device
        self.to(device)

    def forward(self, user, item, **args):
        assert user.shape[0] == user.shape[0]
        return tc.tensor(np.random.uniform(low=0.0, high=1.0, size=(len(user),)))

    def fit(self, *args, **kwrds):
        pass



class PopularRecommend(tc.nn.Module):
    def __init__(self, meta, device, *args, **kwrds):
        tc.nn.Module.__init__(self)
        self.total_freq = 0.1 * len(meta.items)
        self.frequency = [0.1] * len(meta.items)
        self.device = device
        self.to(device)

    def forward(self, user, item, **args):
        proba = [freq / self.total_freq for freq in self.frequency] 
        return tc.tensor([self.frequency[i] / self.total_freq for i in item.flatten().tolist()])

    def fit(self, train, *args, **kwrds):
        for user, item, score, click in train:
            if item >= len(self.frequency):
                self.frequency.extend([0] * (item + 1 - len(self.frequency)))
            value = 1.0 if click else 0.5
            self.frequency[item] = value
            self.total_freq += value

