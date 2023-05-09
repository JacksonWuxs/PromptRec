import random

import torch as tc
import numpy as np
import pandas as pd
import tqdm
from sklearn.metrics import (accuracy_score,
                             roc_auc_score,
                             f1_score,
                             ndcg_score,
                             mean_squared_error,
                             mean_absolute_error,
                             log_loss)



def group_rocauc_score(reals, preds, users):
    ttl_freq, group_auc = 0, 0.0, 
    for uid in set(users.tolist()):
        pos = users == uid
        user_real, user_pred = reals[pos], preds[pos]
        if (user_real == 1.0).sum() in (0, len(user_real)):
            continue
        user_auc = roc_auc_score(user_real, user_pred)
        ttl_freq += len(user_real)
        group_auc += user_auc * len(user_real)
    if ttl_freq == 0.0:
        return float("nan") 
    return group_auc / ttl_freq


class CTREvaluator:
    def __init__(self, data, batch_size, threshold=0.5):
        self.dataset = tc.utils.data.DataLoader(data, batch_size=batch_size,
                                                num_workers=2, shuffle=False)
        self.threshold = threshold
        self.hist_real = []
        self.hist_pred = []
        self.hist_uids = []
        self.hist_iids = []

    def evaluate(self, model, autocast=False):
        model.eval()
        self.clear()
        with tc.no_grad():
            bar = tqdm.tqdm(total=len(self.dataset))
            for batch in self.dataset:
                pred = self._predict(model, batch, autocast)
                self.update(batch["user"], batch["item"], batch["click"], pred)
                bar.update(1)
        model.train()
        return self.score()
    
    def _predict(self, model, batch, autocast):
        if autocast:
            with tc.autocast("cuda"):
                return model(**batch).flatten()
        return model(**batch).flatten()

    def clear(self):
        self.hist_uids.clear()
        self.hist_iids.clear()
        self.hist_pred.clear()
        self.hist_real.clear()

    def update(self, users, items, real_score, pred_score):
        self.hist_uids.extend(self._check_is_python(users))
        self.hist_iids.extend(self._check_is_python(items))
        self.hist_pred.extend(self._check_is_python(pred_score))
        self.hist_real.extend(self._check_is_python(real_score))

    def _check_is_python(self, data):
        if isinstance(data, (tc.Tensor,)):
            data = data.cpu().numpy().flatten().tolist()
        assert all(isinstance(_, (float, int)) for _ in data)
        return data

    def score(self):
        real_score, pred_score = np.array(self.hist_real), np.array(self.hist_pred)
        pred_label = np.where(pred_score >= self.threshold, 1.0, 0.0)
        real_label = np.where(real_score >= self.threshold, 1.0, 0.0)
        return {"auc": roc_auc_score(real_label, pred_score),
                "acc": accuracy_score(real_label, pred_label),
                "gauc": group_rocauc_score(real_score, pred_score, np.array(self.hist_uids)),
                "f1":  f1_score(real_label, pred_label),
                "mse": mean_squared_error(real_score, pred_score),
                "mae": mean_absolute_error(real_score, pred_score),
                "celoss": log_loss(real_score, pred_score)}

