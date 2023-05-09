import itertools
import collections

import torch as tc
import transformers 

from .plms import PLM
from .trainer import BaseTrainer

class PairwiseNSP(tc.nn.Module, BaseTrainer):
    def __init__(self, meta, plm, device, *args, **kwrds):
        tc.nn.Module.__init__(self)
        BaseTrainer.__init__(self)
        self.plm = PLM(plm, "pretrain", device=device) if isinstance(plm, str) else plm
        assert self.plm.head == "pretrain"
        self.sepidx = self.plm.tokenizer.sep_token
        self.clsidx = self.plm.tokenizer.cls_token
        self.device = device
        self.to(device)

    def forward(self, pair_ids, pair_masks, pair_segs, **args):
        return tc.softmax(self.plm(pair_ids, masks=pair_masks, segs=pair_segs)[1], -1)[:, 0].flatten()


class PairwiseContent(tc.nn.Module, BaseTrainer):
    def __init__(self, meta, plm, device, *args, **kwrds):
        tc.nn.Module.__init__(self)
        BaseTrainer.__init__(self)
        self.plm = PLM(plm, "pooler", device=device) if isinstance(plm, str) else plm
        self.device = device
        self._init_embeds(meta, kwrds.get("batchsize", 256))
        self.to(self.device)

    def _init_embeds(self, meta, bs):
        users = map(lambda pair: pair[1], meta.get_profiles("user"))
        items = map(lambda pair: pair[1], meta.get_profiles("item"))
        context = self.plm.encode(itertools.chain(users, items), batchsize=bs, 
                                  total=len(meta.users)+len(meta.items))
        self.uemb = tc.nn.Embedding(len(meta.users), context.shape[-1],
                                    _weight=context[:len(meta.users)])
        self.iemb = tc.nn.Embedding(len(meta.items), context.shape[-1],
                                    _weight=context[len(meta.users):])
        return

    def forward(self, user, item, normalize=True, *args, **kwrds):
        user_emb = self.uemb(user.to(self.device))
        item_emb = self.iemb(item.to(self.device))
        if normalize is True:
            user_emb = tc.nn.functional.normalize(user_emb)
            item_emb = tc.nn.functional.normalize(item_emb)
        dotprod = (user_emb * item_emb).sum(axis=-1)
        if normalize is True:
            return dotprod * 0.5 + 0.5
        return dotprod


class TargetName(tc.nn.Module, BaseTrainer):
    def __init__(self, meta, plm, device, *args, **kwrds):
        tc.nn.Module.__init__(self)
        BaseTrainer.__init__(self)
        self.plm = PLM(plm, "mlm", device=device) if isinstance(plm, str) else plm
        self.target_feature = meta.TargetID
        self.device = device
        self.to(self.device)

    def forward(self, pair_ids, pair_masks, pair_fids, *args, **kwrds):
        index = tc.nonzero(pair_fids == self.target_feature, as_tuple=False)
        logits = self.plm(ids=pair_ids, masks=pair_masks, index=index, return_logits=True)
        begin = 0
        predictions = []
        sizes = collections.Counter(index[:, 0].tolist())
        for sample in range(len(pair_ids)):
            sample_size = sizes[sample]
            sample_logits = logits[begin:begin+sample_size,
                    pair_ids[sample, index[begin:begin+sample_size][:, 1]]]
            predictions.append(sample_logits.mean())
            begin += sample_size
        return tc.stack(predictions, 0)

    def _fit_batch(self, batch):
        pred = self(**batch).flatten()
        real = batch['click'].to(self.device).float().flatten()
        return ((pred - real) ** 2).mean()







        

