import time
import torch as tc
import numpy as np

from transformers import GPT2Tokenizer, GPT2LMHeadModel


def batchit(corpus, size=128):
    assert hasattr(corpus, "__iter__")
    assert size is None or isinstance(size, int) and size > 0
    batch = []
    for row in corpus:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch.clear()
    if len(batch) > 0:
        yield batch



class LogProbComputer(tc.nn.Module):
    def __init__(self, name="gpt2", device="cuda"):
        tc.nn.Module.__init__(self)
        assert "gpt2" in name
        self._name = name
        self._device = device
        self._model = GPT2LMHeadModel.from_pretrained(name, cache_dir="./.cache").to(device)
        self._tokenizer = GPT2Tokenizer.from_pretrained(name, cache_dir="./.cache")
        self._tokenizer.vocab = self._tokenizer.get_vocab()
        self._tokenizer.pad_token = self._tokenizer.cls_token = self._tokenizer.sep_token = self._tokenizer.eos_token
        self._tokenizer.mask_token = "[MASK]"
        self._model.eval()
        self._pad = self._tokenizer.eos_token_id
        self._docs = []

    @property
    def tokenizer(self):
        return self._tokenizer

    def prepare_documents(self, documents):
        self._docs = [self._tokenizer.encode(_) for _ in documents]

    def score(self, cond, batch_size=32, sampling=None):
        cond = self._tokenizer.encode(cond)
        size = len(cond)
        losses = []
        with tc.no_grad():
            for idx, docs in enumerate(batchit(self._docs, batch_size), 1):
                maxlen = max(map(len, docs))
                batch = [cond + _ + [self._pad] * (maxlen - len(_)) for _ in docs]
                batch = tc.tensor(batch).long().to(self._device)
                while True:
                    try:
                        probas = tc.softmax(self._model(batch).logits, -1)
                        break
                    except:
                        time.sleep(1.0)
                for idx, doc in enumerate(docs):
                    logproba = tc.log2(probas[idx, size-1:size+len(doc), tc.tensor(doc)])
                    losses.append(logproba.mean().detach().cpu())
                if idx == sampling:
                    break
        return tc.tensor(losses)


if __name__ == "__main__":
    
    domain = ["Nike football shoes.",
          "New Balance basketball shoes.",
          "Lining trainer shoes.",
          "middle aged female doctor living at New York.",
          "young age male collage student living at New York."]
    corpus = ["Adidas is one of the competitor of Nike.",
          "Nike is a international company producing high quality sport equipments.",
          "Air Jordan is a line of basketball shoes and athletic clothing produced by Nike.",
          "New York is the most popular city in the United States.",
          "Doctors generally wear comfortable sneakers.",
          "The University of Georgia is a public land-grand research university in the southern american.",
          "Gordon Ramsay is a British chef, restaurateur, and television personality. He owned three Michelin restaurants world wide.",
          "Gordon Ramsay is a British chef, restaurateur, television personality. He owned three Michellin restaurants world wide.",
          "AJLKJDSKLJFLSDJFIOUJKLJLKWJOIJKLDSJFLKSDJFKL dklfjlksdJ LKDSJF lj JFlkSJ Flkjoiwejf."
          ]       
    model = LogProbComputer(device="cpu")
    model.prepare_document(domain)
    for d in corpus:
        print(d, model.score(d).mean())

