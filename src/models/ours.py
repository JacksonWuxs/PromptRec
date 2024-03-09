import os
import math
import uuid

import torch as tc
import numpy as np
import transformers as trf
import tqdm

from .plms import PLM


def query_embs(left, right):
    left = tc.nn.functional.normalize(left.mean(dim=0, keepdim=True))
    right = tc.nn.functional.normalize(right)
    corr = left @ right.T
    return sorted(enumerate(corr.cpu().flatten().tolist()), key=lambda p: p[1], reverse=True)


class ScalerOffset(tc.nn.Module):
    def __init__(self, dim):
        tc.nn.Module.__init__(self)
        self._gamma = tc.nn.Parameter(tc.ones(dim), True)
        self._beta = tc.nn.Parameter(tc.zeros(dim), True)

    def forward(self, x):
        return self._gamma * x + self._beta


class TransferSoftPromptRecommander(tc.nn.Module):
    
    def __init__(self, plm, tasks, domains, labels, dropout=0., device="cpu", *args, **kwrds):
        tc.nn.Module.__init__(self)
        
        # general preparation
        self.tasks = tasks
        self.domains = domains
        self.device = device

        self.plm = plm if isinstance(plm, PLM) else PLM(plm)
        self.plm.disable_training()
        self.wordembs = self.plm.encoder.get_input_embeddings()
        print(self.plm.tokenize(" %s" % self.plm.mask_token))
        self.mask_token_id = self.plm.w2i(self.plm.tokenize(" %s" % self.plm.mask_token))[-1]

        # preparation for prompts
        self.sptasks = self._init_embed(tasks[0], tasks[1], tasks[2], True)       # soft prompts of tasks
        self.spdomains = self._init_embed(domains[0], domains[1], domains[2]) # soft prompts of domains
        self.encoder = ScalerOffset(self.plm.dim)
        self.dropout = tc.nn.Dropout(dropout)
                                        
        self._prefix = False
        self._init_labelset(labels)
        self._lossfn = tc.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.to(device)

    @property
    def prefix(self):
        return self._prefix

    @prefix.setter
    def prefix(self, prefix):
        assert isinstance(prefix, bool)
        self._prefix = prefix

    def _init_embed(self, numbers, lengths, hints=None, grad=False):
        wordembs = self.wordembs.weight
        if hints is None:
            corr = query_embs(wordembs[100:5000], wordembs)
            embs = [wordembs[idx].cpu().numpy() for idx, _ in corr[:numbers * lengths]]
            return tc.nn.Parameter(tc.tensor(np.vstack(embs), device=self.device), grad)
        
        assert len(hints) == numbers
        embeds = []
        for hint in hints:
            if hint is None or len(hint) == 0:
                embs = tc.zeros(lengths, wordembs.shape[-1], device=self.device)
                embeds.append(tc.nn.init.kaiming_normal_(embs))
                continue
            
            words = []
            for word in self.plm.tokenize(" " + " ".join(hint)):
                if word not in words:
                    words.append(word)
            if len(words) < lengths:
                with tc.no_grad():
                    embs = self.wordembs(tc.LongTensor(self.plm.w2i(words)).to(self.device))
                    att = query_embs(embs, wordembs)
                    for idx, score in att:
                        word = self.plm.i2w([idx])[0]
                        if word not in words and len(word) >= 2 and word[1:].isalpha():
                            words.append(word)
                            if len(words) == lengths:
                                break
            embeds.append(self.wordembs(tc.LongTensor(self.plm.w2i(words[:lengths])).to(self.device)))
        return tc.nn.Parameter(tc.cat(embeds, axis=0), grad)
                         
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
        
    def forward(self, pair_ids, pair_masks, pair_fids, task, domain, *args, **kwrds):
        ids = pair_ids.to(self.device)
        masks = pair_masks.to(self.device)
        features = pair_fids.to(self.device)

        embs = self.dropout(self.wordembs(ids))
        prefix = [tc.zeros((len(embs), 0, embs.shape[-1]), device=self.device)]
        if self.prefix:
            task, domain = self._get_prefix_prompt(task, domain)
            prefix.append(task)
            prefix.append(self.encoder(domain))
        prefix = tc.cat(prefix, 1)

        # preparing bert inputs
        embs = tc.cat([prefix, embs], 1) # (bs, 2* size + ts, dim)
        masks = tc.cat([tc.ones(prefix.shape[:2], dtype=masks.dtype).to(self.device), masks], -1)
        temp = tc.nonzero(ids == self.mask_token_id, as_tuple=False) + tc.tensor([0, prefix.shape[1]], device=self.device)
        index = []
        bias = 1 if self.plm.head == "clm" else 0
        for idx in range(ids.shape[0]):
            index.append((idx, temp[temp[:, 0] == idx][:, 1].max().item() - bias))
        index = tc.LongTensor(index).to(self.device)
        logits = self.plm(embs=embs, masks=masks, index=index, return_logits=True)
        probs = self._get_verbalize(logits)
        return probs # (bs,)
        
    def _get_prefix_prompt(self, tasks, domains):
        ptasks, pdomains = [], []
        for task, domain in zip(tasks.tolist(), domains.tolist()):
            ptasks.append(self.sptasks[task * self.tasks[1]:(task + 1) * self.tasks[1]])
            pdomains.append(self.spdomains[domain * self.domains[1]:(domain + 1) * self.domains[1]])
        return tc.stack(ptasks), tc.stack(pdomains)

    def _get_verbalize(self, logits):
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
        assert logits.shape[1] == self.wordembs.weight.shape[0]
        #group_logits = [logits[:, group].mean(axis=-1) for group in self.labels]
        group_logits = [logits[:, group].max(axis=-1).values for group in self.labels]
        return tc.softmax(tc.stack(group_logits, dim=1), -1)[:, 1]

    def compute_loss(self, reals, preds, eps=1.0):
        reals = reals.long().to(self.device)
        preds = tc.vstack([1.0 - preds, preds]).T.to(self.device)
        ce_loss = self._lossfn(preds, reals)
        return ce_loss #+ eps * tc.norm(self.sptasks, p=2, dim=-1).mean()

    def _collect_parameters(self, mode="pretrain"):
        assert mode in {"pretrain", "finetune"}, "not supported training mode: %s" % mode
        if mode == "pretrain":
            self.prefix = True
            return [self.sptasks, self.encoder._gamma, self.encoder._beta]
        
        if mode == "finetune":
            params = []
            for n, p in self.plm.named_parameters():
                if "cls" in n and "LayerNorm" not in n:
                    p.requires_grad = True
                    params.append(p)
            print("Training parameters:", len(params))
            return params

    def fit(self, data, bs, lr, epochs=None, steps=None, weight_decay=0.0, decay_rate=0.5, decay_tol=2, early_stop=3, accumulate=1, mode="pretrain", *args, **kwrds):
        data = tc.utils.data.DataLoader(data, batch_size=bs, shuffle=True)
        total_steps = len(data) * epochs if epochs else steps
        epochs = total_steps // len(data) + min(1, total_steps % len(data))       
        scaler = tc.cuda.amp.GradScaler(True)
        parameters = self._collect_parameters(mode)
        optimizer = tc.optim.AdamW(parameters, lr=lr, betas=(0.95, 0.995), weight_decay=weight_decay)
        scheduler = tc.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=decay_rate)

        local = "./outputs/.cache/temp.pth"# % uuid.uuid1()
        tc.save(self.state_dict(), local)
        bar = tqdm.tqdm(total=total_steps)
        best_valid_loss, best_step = -float('inf'), None
        total_tol, current_tol, max_tol = 0, 0, 3
        for epoch in range(1, epochs+1):
            self.train()
            for batch in data:
                bar.update(1)
                with tc.autocast("cuda"):
                    pred = self(**(kwrds | batch)).flatten()
                real = batch["click"].flatten().float()
                loss = self.compute_loss(real, pred, weight_decay)
                scaler.scale(loss / accumulate).backward()
                if (bar.n + 1) % accumulate == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                if total_steps == bar.n:
                    break
            if kwrds.get("valid"):
                results = kwrds["valid"].evaluate(self, True)
                valid_loss = results["gauc"]
                print("Epoch=%d | Valid=%.5f | Best=%.5f" % (epoch, valid_loss, best_valid_loss))
                if valid_loss >= best_valid_loss + 1e-5:
                    best_valid_loss, best_step = valid_loss, bar.n
                    current_tol = 0
                    tc.save(self.state_dict(), local)
                else:
                    current_tol += 1
                    if current_tol == decay_tol:
                        print("Reducing learning rate!")
                        scheduler.step()
                        self.load_state_dict(tc.load(local))
                        total_tol += 1
                        current_tol = 0
                    if total_tol == early_stop:
                        print("Early stop at epoch %s with best gauc %.4f." % (epoch, best_valid_loss))
                        break
        if best_step:
            print("Reload weights: Step=%d | Valid-GAUC=%.4f" % (best_step, best_valid_loss))
            self.load_state_dict(tc.load(local))
            os.remove(local)
        return self




