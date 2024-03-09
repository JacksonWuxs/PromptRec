import os
import tqdm
import numpy as np
import torch as tc
import transformers as trf

os.environ["TOKENIZERS_PARALLELISM"] = "false" 
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = tc.argmax(tc.abs(u), 0)
    i = tc.arange(u.shape[1]).to(u.device)
    signs = tc.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v


class PLM(tc.nn.Module):
    def __init__(self, name, head, tokenizer=None, device="cpu", load_in_8bit=False):
        super(PLM, self).__init__()
        assert head in ("mlm", "clm", "pretrain", "max", "mean", "pooler", "cls", "s2s")
        self.device = device
        self.head = head
        self._name = name + head
        auto_model = {"mlm": trf.AutoModelForMaskedLM,
                      "clm": trf.AutoModelForCausalLM,
                      "s2s": trf.AutoModelForSeq2SeqLM,
                      "pretrain": trf.AutoModelForPreTraining,\
                      }.get(head, trf.AutoModel)
        self.encoder = auto_model.from_pretrained(name)
        self.dim = self.encoder.config.hidden_size
        self.tokenizer = tokenizer if tokenizer else PLM.load_tokenizer(name)
        self.tokenizer.pad_token = self.tokenizer.eos_token = "<|endoftext|>"
        self.padid = self.w2i(self.tokenizer.pad_token) if self.tokenizer.pad_token else 0
        for names in [("pad", "eos"), ("cls", "bos"), ("sep", "eos"), ("mask",)]:
            token = ""
            for name in names:
                if hasattr(self.tokenizer, name + "_token"):
                    token = getattr(self.tokenizer, name + "_token")
                    break
            setattr(self, name + "_token", token)
        self.to(self.device)

    @classmethod
    def load_tokenizer(cls, name):
        return trf.AutoTokenizer.from_pretrained(name)

    def disable_training(self):
        self.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
            param.requires_grad_(False)

    def w2i(self, tokens):
        if len(tokens) == 0:
            return []
        if isinstance(tokens, str):
            return self.w2i([tokens])[0]
        if not isinstance(tokens[0], str):
            return [self.w2i(_) for _ in tokens]
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def i2w(self, ids):
        if len(ids) == 0:
            return []
        if not isinstance(ids[0], int):
            return [self.i2w(_) for _ in ids]
        return self.tokenizer.convert_ids_to_tokens(ids)

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def prepare4generate(self, texts):
        inputs = self.tokenizer(texts, padding=True, return_tensors="pt")
        batch_size, seq_len = inputs["input_ids"].shape
        inputs["attention_mask"] = tc.flip(inputs["attention_mask"], dims=[1])
        shifts = seq_len - inputs["attention_mask"].sum(dim=-1)
        for idx in range(batch_size):
            inputs["input_ids"][idx] = inputs["input_ids"][idx].roll(shifts[idx].item())
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _prepare(self, ids, embs, segs, masks):
        if ids is not None and ids.device != self.device:
            ids = ids.to(dtype=tc.long, device=self.device, non_blocking=True)
        if embs is not None and embs.device != self.device:
            embs = embs.to(dtype=tc.float32, device=self.device, non_blocking=True)
        if masks is not None and masks.device != self.device:
            masks = masks.to(dtype=tc.long, device=self.device, non_blocking=True)
        elif ids is not None:
            masks = tc.where(ids != self.padid, 1.0, 0.0).to(dtype=tc.long,device=self.device, non_blocking=True)
        elif embs is not None:
            masks = tc.where(tc.sum(embs.abs(), -1) > 0, 1.0, 0.0).to(dtype=tc.long, device=self.device, non_blocking=True)
        if "t5" in self._name:
            decoder_ids, decoder_embs = None, None
            if ids is not None:
                decoder_ids = self.encoder._shift_right(ids)
            if embs is not None:
                decoder_embs = embs.new_zeros(embs.shape)
                decoder_embs[:, 1:] = embs[:, :-1].clone()
                decoder_embs[:, 0] = self.encoder.get_input_embeddings()(tc.LongTensor([self.encoder.config.decoder_start_token_id]).to(self.device))
            return {"input_ids": ids,
                    "decoder_input_ids": decoder_ids, 
                    "inputs_embeds": embs, 
                    "decoder_inputs_embeds": decoder_embs,
                    "attention_mask": masks}
        if "dis" in self._name:
            return {"inputs_embeds": embs,
                    "attention_mask": masks}
        if segs is not None and segs.device != self.device:
            segs = segs.to(dtype=tc.long, device=self.device, non_blocking=True)
        return {"input_ids": ids,
                "inputs_embeds": embs,
                "token_type_ids": segs,
                "attention_mask": masks}

    def _postprocess(self, inputs, outputs, index, return_logits):
        if self.head == "pretrain":
            return outputs

        if self.head == "pooler":
            outputs = outputs[1]
            if not isinstance(outputs, tc.Tensor):
                raise RuntimeError("Current pretrain model doesn't support pooler")
            return outputs

        if self.head == "cls":
            return outputs[0][:, 0, :]

        if self.head == "mean":
            embs, mask = outputs[0], inputs["attention_mask"]
            embs = embs * mask.unsqueeze(-1)
            return tc.sum(embs, 1) / (mask.sum(1, keepdim=True) + 1e-9)

        if self.head == "max":
            return tc.max(outputs[0], 1).values

        if self.head == "mlm":
            if index is None:
                assert inputs["input_ids"] is not None
                index = tc.nonzero(inputs["input_ids"] == self.tokenizer.mask_token_id, as_tuple=False).to(self.device)
            logits = outputs[0][index[:, 0], index[:, 1]].squeeze()
            
        elif self.head in ("clm", "s2s"):
            logits = outputs[0][:, -1, :]
        
        if return_logits:
            return logits
        return tc.softmax(logits, -1)

    def _encode(self, ids=None, embs=None, segs=None, masks=None, index=None, return_logits=True):
        inputs = self._prepare(ids, embs, segs, masks)
        mask = inputs["attention_mask"]
        inputs["return_dict"] = True
        if "bert" in self._name.lower():
            embs = self.encoder(**inputs).last_hidden_state
            return (embs * mask.unsqueeze(-1)).sum(axis=1) / (mask.sum(axis=1, keepdims=True) + 1e-7)
        elif "t5" in self._name.lower():
            del inputs["decoder_input_ids"], inputs["decoder_inputs_embeds"]
            embs = self.encoder.encoder(**inputs).last_hidden_state
            return (embs * mask.unsqueeze(-1)).sum(axis=1) / (mask.sum(axis=1, keepdims=True) + 1e-7)
        elif "gpt" in self._name.lower():
            embs = self.encoder(**inputs).last_hidden_state
            return (embs * mask.unsqueeze(-1)).sum(axis=1) / (mask.sum(axis=1, keepdims=True) + 1e-7)
        raise NotImplementedError("Unknow model for encoding")
        
                
    def forward(self, ids=None, embs=None, segs=None, masks=None, index=None, return_logits=True):
        inputs = self._prepare(ids, embs, segs, masks)
        outputs = self.encoder(**inputs)
        return self._postprocess(inputs, outputs, index, return_logits)

    def encode(self, texts, batchsize=64, maxlen=510, total=None, whiten=True):
        cls = self.tokenizer.cls_token if self.tokenizer.cls_token else self.tokenizer.bos_token
        sep = self.tokenizer.sep_token if self.tokenizer.sep_token else self.tokenizer.eos_token
        if cls is None:
            cls = ""
        if sep is None:
            sep = ""
        if maxlen is None:
            maxlen = self.tokenizer.model_max_length
        if total:
            bar = tqdm.tqdm(total=total // batchsize)
        embs, batch = [], []
        with tc.no_grad():
            self.eval()
            for ids in texts:
                if isinstance(ids, str):
                    ids = cls + " " + ids + " " + sep
                    ids = self.w2i(self.tokenize(ids))
                batch.append(ids[:maxlen])
                if len(batch) == batchsize:
                    batch_maxlen = max(map(len, batch))
                    for sample in batch:
                        sample.extend([self.padid] * (batch_maxlen - len(sample)))
                    inputs = tc.tensor(batch, device=self.device).reshape((len(batch), -1)).long()
                    embs.append(self._encode(inputs))
                    batch.clear()
                    if total:
                        bar.update(1)
            if len(batch) > 0:
                batch_maxlen = max(map(len, batch))
                for sample in batch:
                    sample.extend([self.padid] * (batch_maxlen - len(sample)))
                inputs = tc.tensor(batch, device=self.device)
                embs.append(self._encode(inputs.reshape((len(batch), -1))))
            if whiten:
                return self._whitening(tc.cat(embs, axis=0).to(self.device))
            return tc.cat(embs, axis=0)
        
    def _whitening(self, embs):
        # PCA
        mu = tc.mean(embs, dim=0, keepdim=True)
        X = embs - mu
        U, S, V = tc.svd(X)
        U, Vt = svd_flip(U, V)
        accumulate, sum_S = 0.0, sum(S.detach().cpu().tolist())
        for idx, s in enumerate(S.detach().cpu().tolist(), 1):
            accumulate += s / sum_S
            if accumulate > 0.8:
                break
        X = tc.mm(X, Vt[:idx].T)
        
        # whitening
        u, s, vt = tc.svd(tc.mm(X.T, X) / (X.shape[0] - 1.0))
        W = tc.mm(u, tc.diag(1.0 / tc.sqrt(s)))
        X = tc.mm(X, W)
        return X

    
    

if __name__ == "__main__":
    sentences = ["this is a test case [MASK]", "this is the user case [MASK]"]
    plm = PLM("gpt2", head=None)
    ids = [plm.tokenizer.convert_tokens_to_ids(plm.tokenizer.tokenize(_)) for _ in sentences]
    
    print(plm.encode(ids, total=2).shape)

