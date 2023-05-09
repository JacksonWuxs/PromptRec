import os
import math
import uuid

import torch as tc
import numpy as np
import transformers as trf
import tqdm

from .plms import PLM



class PromptRecommander(tc.nn.Module):
    
    def __init__(self, plm, labels, device="cpu", *args, **kwrds):
        tc.nn.Module.__init__(self)
        self.plm = plm if isinstance(plm, PLM) else PLM(plm)
        self.plm.disable_training()
        self.wordembs = self.plm.encoder.get_input_embeddings()
        self.mask_token_id = self.plm.w2i(self.plm.tokenize(" %s" % self.plm.mask_token))[-1]
        self.device = device
        self._init_labelset(labels)
        self.to(device)
                         
    def _init_labelset(self, labels):
        assert isinstance(labels, (list, tuple, dict)) and len(labels) == 2
        if isinstance(labels, dict):
            assert 0 in labels and 1 in labels
            labels = [labels[0], labels[1]]
        labels = labels.copy()
        for idx in range(2):
            tmp = []
            for key in labels[idx]:
                key_idx = self.plm.w2i([key])[0] if key in self.plm.tokenizer.vocab else float("inf")
                key_tmp = self.plm.tokenize(" " + key)[-1]
                if self.plm.w2i(key_tmp) < key_idx:
                    tmp.append(key_tmp)
                else:
                    tmp.append(key)
            labels[idx] = self.plm.w2i(tmp)
        self.labels = [tc.LongTensor(group).to(self.device) for group in labels]        
        
    def forward(self, pair_ids, pair_masks, pair_fids, *args, **kwrds):
        ids = pair_ids.to(self.device)
        masks = pair_masks.to(self.device)

        embs = self.wordembs(ids)
        prefix = tc.zeros((len(embs), 0, embs.shape[-1]), device=self.device)

        # preparing bert inputs
        embs = tc.cat([prefix, embs], 1) 
        masks = tc.cat([tc.ones(prefix.shape[:2], dtype=masks.dtype).to(self.device), masks], -1)
        index = tc.nonzero(ids == self.mask_token_id, as_tuple=False) + tc.tensor([0, prefix.shape[1]], device=self.device)
        logits = self.plm(embs=embs, masks=masks, index=index, return_logits=True)
        probs = self._get_verbalize(logits)
        return probs # (bs,)

    def _get_verbalize(self, logits):
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
        assert logits.shape[1] == self.wordembs.weight.shape[0]
        group_logits = [logits[:, group].mean(axis=-1) for group in self.labels]
        return tc.softmax(tc.stack(group_logits, dim=1), -1)[:, 1]
