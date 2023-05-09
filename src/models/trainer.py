import uuid
import os
import torch as tc
import transformers as trf
import tqdm


class BaseTrainer:
    def __init__(self, workdir="./.cache/"):
        os.makedirs(workdir, exist_ok=True)
        self._key = str(uuid.uuid1())
        self._fpath = workdir + self._key + ".pth"
        self._lossfn = tc.nn.BCELoss()

    def _get_batch(self, dataset):
        for batch in dataset:
            yield batch

    def _fit_batch(self, batch):
        return self._lossfn(self(**batch).flatten(),
                           batch['click'].float().flatten().to(self.device))

    def fit(self, train, lr, bs, valid=None, epochs=None, steps=None, *args, **kwrds):
        assert not (epochs and steps)
        train = tc.utils.data.DataLoader(train, batch_size=bs, shuffle=True)
        if valid:
            valid = tc.utils.data.DataLoader(valid, batch_size=bs, shuffle=False)
        if epochs is None:
            epochs = steps // len(train)
        else:
            steps = len(train) * epochs
        optim = tc.optim.SGD(self.parameters(), lr=lr, weight_decay=kwrds.get("wd", 0.0), momentum=0.9)
        
        bar = tqdm.tqdm(total=steps)
        best_loss, best_epoch = float('inf'), 0
        for epoch in range(1, epochs + 1):
            self.train()
            total_train_loss = 0.0
            for batch in self._get_batch(train):
                optim.zero_grad()
                loss = self._fit_batch(batch)
                total_train_loss += loss.cpu().item()
                loss.backward()
                optim.step()
                bar.update(1)

            if valid:
                self.eval()
                valid_loss = 0.0
                for batch in self._get_batch(valid):
                    valid_loss += self._fit_batch(batch).cpu().item()
                if valid_loss < best_loss:
                    best_loss, best_epoch = valid_loss, epoch
                    tc.save(self.state_dict(), self._fpath)
        if valid:
            self.load_state_dict(tc.load(self._fpath))
            os.remove(self._fpath)
            
        
