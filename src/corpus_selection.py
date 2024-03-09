import re
import os
import sys
import time
import math
import functools

import nltk
import torch as tc
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder

from models.plms import PLM
from models.quick_LM_evaluate import LogProbComputer
from datautils.core.corpus import PublicCorpus, CorpusSearchIndex
from datautils.movielens import MovieLensMeta
from datautils.coupons import CouponsMeta
from datautils.restaurants import RestaurantMeta


SEED = int(sys.argv[1])
CUDA = str(sys.argv[2])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA 
#tc.cuda.set_device(int(CUDA))


class BaseCorpus:
    def __init__(self, tokenizer, maxlen=510, num_worker=2):
        self.plm = tokenizer
        self.maxlen = maxlen
        self.eos_id = tokenizer.convert_tokens_to_ids([tokenizer.cls_token])[0]
        self.sep_id = tokenizer.convert_tokens_to_ids([tokenizer.sep_token])[0]
        self.mask_id = tokenizer.convert_tokens_to_ids([tokenizer.mask_token])[0]

    @property
    def maxlen(self):
        return self._maxlen

    @maxlen.setter
    def maxlen(self, newlen):
        assert isinstance(newlen, int)
        self._maxlen = newlen
        self._halflen = newlen // 2

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]        

    def _prepare_document(self, doc):
        assert doc is None or isinstance(doc, (list, tuple, str))
        if doc is None:
            return None
        
        if isinstance(doc, str):
            return self.plm.convert_tokens_to_ids(self.plm.tokenize(doc))
        
        assert all(map(lambda x: isinstance(x, (int, str)), doc))
        if isinstance(doc, str):
            return self.plm.convert_tokens_to_ids(doc)
        return doc

    def get_batches(self, bs, device="cpu"):
        tensor = functools.partial(tc.tensor, device=device)
        batch = []
        for sample in self:
            batch.append(sample)
            if len(batch) == bs:
                yield tuple(map(tensor, zip(*batch)))
                batch.clear()
        if len(batch) > 0:
            yield tuple(map(tensor, zip(*batch)))


class SingleDocumentMask(BaseCorpus):
    def __init__(self, tokenizer, doc, maxlen=510):
        if isinstance(doc, str):
            doc = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(doc))
        self.doc = tokenizer.decode(doc)
        self.ids = self._prepare_document(doc)
        BaseCorpus.__init__(self, tokenizer, min(maxlen, len(self.ids)))
        self.masks = [1] * (self._maxlen + 2)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        word = self.ids[idx]
        self.ids[idx] = self.mask_id
        ids = [self.eos_id] + self.ids.copy() + [self.sep_id]
        self.ids[idx] = word
        return ids, self.masks, idx + 1, word


class PairDocumentRetrival(BaseCorpus):
    def __init__(self, tokenizer, documents, maxlen=509):
        BaseCorpus.__init__(self, tokenizer, maxlen)
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
        self.docs = documents
        self.cands = documents
        self.embs = self.encode(documents).T
        self.qry = None

    @property
    def query(self):
        return self.qry

    @query.setter
    def query(self, new_query):
        self.qry = self._prepare_document(new_query)

    def encode(self, texts):
        if not isinstance(texts, list):
            texts = list(texts)
        return self.model.encode(texts, device="cuda", convert_to_tensor=True, normalize_embeddings=False)
        return tc.nn.functional.normalize(embeds, axis=-1)

    def remove_duplicates(self):
        cand_hash, new_cands = set(), []
        for cand in self.cands:
            if cand not in cand_hash:
                cand_hash.add(cand)
                new_cands.append(cand)
        self.cands = new_cands

    def estimate_threshold(self, size=5000):
        def nonzero_mean(x):
            total = x.sum()
            nonzero = x.nonzero().shape[0]
            return total / nonzero
        dataset = load_dataset("sedthh/gutenberg_english", "train")["train"].shuffle()
        dataset = self.encode([x["TEXT"][:1000] for i, x in enumerate(dataset) if i <= size])
        self.unmatch = nonzero_mean(tc.tril(dataset @ self.embs)).item()
        self.matched = nonzero_mean(tc.tril(self.embs.T @ self.embs, diagonal=-1)).item()
        self.ranged = self.matched - self.unmatch
        self.threshold = ((self.unmatch + self.matched) / 2)
        print("Estimated threshold: %.4f" % self.threshold, self.unmatch, self.matched)

    def predict(self, query):
        query = self.encode([query])
        similar = (query @ self.embs).squeeze() 
        logit = 4 * ((similar - self.unmatch) / self.ranged) - 2.0
        return 1. / (1. + tc.exp(-logit))

    def pretokenize(self):
        self.cands = [self.plm.convert_tokens_to_ids(self.plm.tokenize(_)) for _ in self.cands]

    def cand_length(self, percent=1.0):
        assert isinstance(percent, float) and 0 < percent <= 1.0
        return sorted(map(len, self.cands))[max(0, min(len(self.cands) - 1, int(len(self.cands) * percent)))]

    def __len__(self):
        return len(self.cands)

    def __getitem__(self, idx):
        query, candidate = self.qry, self.cands[idx]
        if len(query) + len(candidate) > self._maxlen:
            query = self.qry[:self._maxlen - len(candidate)]
        ids = [self.eos_id] + query + [self.sep_id] + candidate + [self.sep_id]
        padding = [0] * (3 + self._maxlen - len(ids))
        return (ids + padding,
                [1] * len(ids) + padding, # masks
                [0] * (2 + len(query)) + [1] * (1 + len(candidate)) + padding) # segs

    def _truncate_pair(self, seq_a, seq_b):
        while len(seq_a) + len(seq_b) > self._maxlen:
            if len(seq_a) > len(seq_b):
                seq_a.pop()
            else:
                seq_b.pop()


def document_probability(plm, document, seqlen, bs=32):
    probas = []
    for ids, masks, segs, pos, wid in document.get_batches(bs, plm.device):
        proba = tc.softmax(plm(ids=ids, masks=masks, segs=segs)[0], -1)
        probas.append(proba[tc.arange(len(proba)), pos.long(), wid.long()])
    return 2 ** tc.log2(tc.cat(probas)).mean()


def pairwise_probability(plm, query, candidates, bs=32):
    return candidates.predict(query).float()

def mutual_info(plm, query, candidates, bs):
    pair_probas = candidates.predict(query).float()
    if pair_probas.mean().item() <= 0.5:
        return pair_probas.mean().item(), 0.0, -1e20
    cond_probas = plm.score(query, 256, sampling=4)
    mi = (pair_probas[:len(cond_probas)].cpu() * cond_probas.cpu()).sum()
    return pair_probas.mean().item(), cond_probas.mean().item(), mi.item()


def refine_corpus(plm, corpus, domain, corpus_maxlen=128, batchsize=1024):
    tokenizer = plm.tokenizer
    plm.prepare_documents(domain)
    if not isinstance(domain, PairDocumentRetrival):
        domain = PairDocumentRetrival(tokenizer, domain, 510)
    domain.remove_duplicates()
    domain.pretokenize()
    domain.estimate_threshold()
    domain_maxlen = domain.cand_length(1.0)

    splitter = re.compile(r'(?<=[.!?:])\s+') 
    with tc.no_grad():
        plm.bfloat16()
        #plm.disable_training()
        for document in corpus:
            # Rule-0: each document must be long enough
            if len(document) <= 200:
                continue
            if len(set(document.split(" "))) <= 30:
                continue
            # Rule-1: each sentence must has at least 10 characters
            sentences = [_ for _ in splitter.split(document, 10) if len(_) > 10]
            if len(sentences) <= 2:
                continue
            heads =  " ".join(" ".join(sentences[:8]).split(" ")[:corpus_maxlen])
            pair, single, score = mutual_info(plm, heads, domain, batchsize)
            yield (single, pair, score, document.replace("\n", " "))


def selecting(src, tgt, topK=1.0, seg="\t"):
    src = CorpusSearchIndex(src)
    scores = sorted(enumerate(map(lambda x: float(x.split(seg, 3)[1]), src)),
                    reverse=True, key=lambda pair: pair[1])
    if isinstance(topK, float):
        topK = int(len(scores) * topK)
    for temp, (idx, val) in enumerate(scores, 1):
        if val <= 1e-10:
            break
    print(temp, "warning!")
    print(np.mean([_[1] for _ in scores]))
    print("10%", scores[int(0.1 * len(scores))][1])
    print("20%", scores[int(0.2 * len(scores))][1])
    print("50%", scores[int(0.5 * len(scores))][1])
    current = 0
    with open(tgt, "w", encoding="utf8") as tgt:
        for _, (idx, score) in enumerate(scores, 1):
            single, pair, mutual, text = src[idx].split("\t", 3)
            #tgt.write(single +'\t' + pair +'\t'+ mutual + '\t' + text + "\n")
            tgt.write(text + "\n")
            current += 1
            if topK == current:
                break
    print("Totally collect %d documents by scaning %d." % (current, _)) 
            

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
root = r"../datasets/refined_corpus/"
data = "../datasets/downstream_tasks/%s/"
os.makedirs(root, exist_ok=True)
save = root + "c4_%s_%sto%s_skip%s.txt"
device = "cuda:%s" % CUDA

if __name__ == "__main__":
    tokenizer = PLM("prajjwal1/bert-small", "pretrain", device="cuda").tokenizer
    plm = LogProbComputer("gpt2")


    if sys.argv[3].count("-") == 2:
        dataname = sys.argv[4]
        suffix = "In short, the user feels [::MASK::] about the item."
        MetaClass = {"coupon": CouponsMeta, "mexico_restaurant": RestaurantMeta, "ml-100k": MovieLensMeta}[dataname]
        start, stop, skip = list(map(int, sys.argv[3].split("-")))
        meta = MetaClass(data % dataname, tokenizer, suffix=suffix)
        domain = set(meta.exhaustive_sampling([["negative"], ["positive"]], fullsize=len(meta.items) * 10))
        print("Total Number of Domain Documents: %d" % len(domain))
        corpus = save % (dataname, start, stop, skip)
        with open(corpus, "a+", encoding="utf8") as fout:
            idx = 0
            fout.seek(0)
            for idx, row in enumerate(fout):
                pass
            start = idx * skip + start
            corpus = PublicCorpus(("c4", "en"), "train", "text", start, stop, skip)
            for _ in refine_corpus(plm, corpus, domain):
                fout.write("\t".join(map(str, _)) + "\n")
    else:
        src, topK = sys.argv[3], int(sys.argv[4])
        folder, file = os.path.split(sys.argv[3])
        tgt = folder + "/top%s_" % topK + file
        selecting(src, tgt, topK)
    

