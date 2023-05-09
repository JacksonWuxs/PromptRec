import re
import time
import random
import string

from nltk.corpus import stopwords
import numpy as np
import transformers as trfm


class Tokenizer:
    def __init__(self, pretrain, max_word=None, keep_head_rate=0.7, drop_stopword=True, drop_uncommen=True):
        self.max_word = max_word if max_word else float('inf')
        self.keep_head = keep_head_rate
        self.dropstopword = drop_stopword
        self.dropuncommen = drop_uncommen
        self.stopwords = set(stopwords.words("english")) | set(string.digits + string.punctuation)
        self.segwords = set(string.punctuation)
        self.tokenizer = trfm.BertTokenizer.from_pretrained(pretrain, cache_dir="./cache")
        self.vocabs = set(self.tokenizer.get_vocab())

    def encode(self, text):
        text = self.clean(text)
        tokens = self.tokenize(text)
        return self.transform(tokens)

    def preprocess(self, text):
        text = self.clean(text)
        return self.tokenize(text)

    def transform(self, tokens):
        tokens = ["[CLS]"] + self.truncate(tokens) + ["[SEP]"]
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        mask = self.masking(ids)
        ids = self.padding(ids)
        assert len(mask) == len(ids)
        return ids, mask
    
    def truncate(self, tokens):
        if len(tokens) <= self.max_word:
            return tokens
        split = int(self.max_word * self.keep_head)
        return tokens[:split] + tokens[len(tokens) - (split - self.max_word):]

    def padding(self, ids):
        ids.extend([0] * (2 + self.max_word - len(ids)))
        return np.array(ids).astype(np.int32)

    def masking(self, ids):
        mask = [1] * len(ids) + [0] * (2 + self.max_word - len(ids))
        return np.array(mask).astype(np.int32)

    def clean(self, text):
        text = re.sub("<.*?>", "", text) # remove html tags
        text = re.sub("&[a-z0-9]+|&#[0-9]{1,6}|&#x[0-9a-f]{1,6}", "", text) # remove HTML Entities
        text = re.sub("www\S+", "", text) # remove URLs
        text = re.sub("http\S+", "", text) # remove URLs
        for pattern, rephrase in [(" won\'t ", "will not"),
                                  (" can\'t ", "can not"),
                                  ("n\'t ", " not "),
                                  ("\'re ", " are "),
                                  ("\'s", " is "),
                                  ("\'d", " would "),
                                  ("\'ll", " will "),
                                  ("\'t", " not "),
                                  ("\'ve", " have "),
                                  ("\'m", " am "),
                                  (" n't ", " not ")]:
            text = text.replace(pattern, rephrase)
        text = re.sub("[^A-Za-z0-9',.!?]", " ", text)
        text = re.sub(r"!+", "!", text)
        text = re.sub(r"\.+", ".", text)
        text = re.sub(r"\?+", "?", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.replace("' s", "'s")

    def tokenize(self, text):
        tokens = []
        for token in self.tokenizer.tokenize(text):
            if self.dropstopword and token in self.stopwords:
                continue
            if len(token) > 0 and token != "\\":
                tokens.append(token)
        return tokens

    def i2w(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)

    def w2i(self, words):
        return self.tokenizer.convert_tokens_to_ids(words)
            
