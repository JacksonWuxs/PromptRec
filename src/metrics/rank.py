import random

import tqdm
import numpy as np
import torch as tc
from collections import defaultdict

from sklearn.metrics import ndcg_score



class TestDataset(tc.utils.data.Dataset):
    def __init__(self, uids, iids, meta_index, clicked_history):
        self.uids = uids
        self.iids = iids
        self.meta = meta_index
        self.user_click = clicked_history

    def __len__(self):
        return len(self.iids) * len(self.uids)

    def __getitem__(self, idx):
        uid = idx // len(self.iids)
        iid = idx % len(self.iids)
        return self.meta.get_feed_dict(self.uids[uid], self.iids[iid])

    
class RankEvaluator:
    def __init__(self, meta, train, valid, k, batch_size=128, num_test=None):
        self._batchsize = batch_size
        self.meta = meta
        self.top_k = k
        self.test_data = self._prepare_test_data(train, valid, num_test)

    def _prepare_test_data(self, train, valid, num_test):
        train = self._combine_records(train)
        valid = self._combine_records(valid)
        items = list({__ for _ in train.values() for __ in _} &\
                     {__ for _ in valid.values() for __ in _})
        users = list(set(train.keys()) & set(valid.keys()))
        users = [uid for uid in users if len(valid[uid]) > 0]
        if num_test is not None:
            users = random.sample(users, k=num_test)
        return TestDataset(users, items, self.meta, valid)

    def _combine_records(self, data):
        records = defaultdict(set)
        if data is None:
            return records
        for uid, iid, rate, click in data:
            if click == 1.0:
                records[uid].add(iid)
        return records
    
    def _predict(self, model, batch, autocast=False):
        if autocast:
            with tc.autocast("cuda"):
                return model(**batch).cpu().detach().flatten().tolist()
        return model(**batch).cpu().detach().flatten().tolist()
    
    def evaluate(self, model, autocast=False):
        hitrate = {k: [] for k in self.top_k}
        precision = {k: [] for k in self.top_k}
        recall = {k: [] for k in self.top_k}
        ndcg = {k: [] for k in self.top_k}
        model.eval()
        with tc.no_grad():
            users, items, scores = [], [], []
            num_item = len(self.test_data.iids)
            dataset = tc.utils.data.DataLoader(self.test_data, batch_size=self._batchsize,
                                               shuffle=False, pin_memory=True)
            for batch in tqdm.tqdm(dataset):
                scores.extend(self._predict(model, batch, autocast))
                items.extend(batch["item"].flatten().tolist())
                users.extend(batch["user"].flatten().tolist())
                
                while len(scores) >= num_item:
                    user_clicked = self.test_data.user_click[users[0]]
                    item_sorted, score_sorted = [], []
                    for score, item in sorted(zip(scores[:num_item], items[:num_item]), reverse=True):
                        item_sorted.append(item)
                        score_sorted.append(score)
                    hits = [1.0 if iid in user_clicked else 0.0 for iid in item_sorted]
    
                    for k in self.top_k:
                        num_hit = sum(hits[:k])
                        hitrate[k].append(1.0 if num_hit >= 1 else 0.0)
                        precision[k].append(num_hit / k)
                        recall[k].append(num_hit / len(user_clicked))
                        ndcg[k].append(ndcg_score([hits], [score_sorted], k=k))
                    users, items, scores = users[num_item:], items[num_item:], scores[num_item:]
                        
        model.train()
        assert len(hitrate[self.top_k[0]]) == len(self.test_data.uids)
        return {"top_k": self.top_k,
                "precision": [np.mean(precision[k]) for k in self.top_k],
                "recall": [np.mean(recall[k]) for k in self.top_k],
                "ndcg": [np.mean(ndcg[k]) for k in self.top_k],
                "hit": [np.mean(hitrate[k]) for k in self.top_k]
                } 

    
