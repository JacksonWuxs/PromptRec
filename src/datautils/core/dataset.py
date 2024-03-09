import os
import random
import itertools
import functools

import numpy as np
import torch as tc
import tqdm

from .lookup import LookupTable
from .corpus import CorpusSearchIndex
from .templates import Template
from .verbalizers import Simple, Continue
from .cache import HashCache


class _BaseType:
    def __init__(self, root, category, split, verbalizers, template):
        assert os.path.exists(root)
        assert isinstance(split, str)
        assert isinstance(verbalizers, (list, tuple))
        assert all(isinstance(_, Simple) for _ in verbalizers)
        assert isinstance(template, Template)
        self._root = root
        self._category = category
        self._segment = split
        self._template = template
        self._verbalizers = verbalizers
        
        self._names = LookupTable.from_txt(root + r"/%s_idx.txt" % category)
        self._meta = CorpusSearchIndex(root + r"/%s_meta.txt" % category)
        self._cache_verb = HashCache(self._verbalize, len(self._names), "freq")
        self._cache_vect = HashCache(self._vectorize, len(self._names), "freq")
        assert len(self._names) == len(self._meta)

    def __len__(self):
        return len(self._names)

    def _get_feature(self, idx):
        assert isinstance(idx, int) and idx < len(self._names)
        features = self._meta[idx].split(self._segment)
        #assert len(features) == len(self._verbalizers)
        return features

    def get_sample_index(self, instance_name):
        if isinstance(instance_name, str):
            return self._names[instance_name]
        return instance_name

    def get_features(self, idx, use_cache=True):
        if use_cache:
            return self._cache_verb.collect(idx)
        return self._verbalize(idx)

    def get_profile(self, idx, use_cache=True):
        features = self.get_features(idx, use_cache)
        return self._template.construct(**features)[:2]

    def get_vector(self, idx, use_cache=True):
        if use_cache:
            return self._cache_vect.collect(idx)
        return self._cache_vect(idx)

    def list_features(self, use_cache=True):
        counts = {verb.name: set() for verb in self._verbalizers}
        for idx in range(len(self)):
            for fname, fval in self.get_features(idx).items():
                if fval is not None:
                    counts[fname].add(fval)
        return [{"name": verb.name,
                 "continue": isinstance(verb, Continue),
                 "choices": counts[verb.name],
                 "length": len(counts[verb.name]),
                 "miss": verb.miss,
                 "side": self._category} \
                for verb in self._verbalizers]

    def _verbalize(self, idx):
        features = self._get_feature(idx)
        return {verb.name: verb.verbalize(feat) for feat, verb in zip(features, self._verbalizers)}

    def _vectorize(self, idx):
        features = self._get_feature(idx)
        return [verb.vectorize(feat) for feat, verb in zip(features, self._verbalizers)]
        

class BaseMeta:
    def __init__(self, root, tokenizer, split,
                 user_verb, item_verb, user_str, item_str, prefix, suffix, slots, maxlen, target_name,
                 task=0, domain=0,
                 return_profile=True, return_vector=True, include_sep=False):
        assert isinstance(task, int) and isinstance(domain, int)
        self._return_profile = return_profile
        self._return_vector = return_vector
        self.root = os.path.abspath(root).replace(r"\\", "/")
        self.task, self.domain = task, domain
        cls = tokenizer.cls_token if tokenizer.cls_token else ""
        sep = tokenizer.sep_token if tokenizer.sep_token else ""
        self._tokenizer = tokenizer
        
        pair_str = cls + prefix + user_str
        if include_sep is True:
            pair_str += sep
        pair_str += item_str + sep + suffix
        self.pair_temp = Template(prompt=pair_str, tokenizer=tokenizer, maxlen=maxlen, slot_args=slots)
        self.user_temp = Template(prompt=cls + user_str, tokenizer=tokenizer, maxlen=maxlen, slot_args=slots)
        self.item_temp = Template(prompt=cls + item_str, tokenizer=tokenizer, maxlen=maxlen, slot_args=slots)
        self.users = _BaseType(self.root, "user", split, user_verb, self.user_temp)
        self.items = _BaseType(self.root, "item", split, item_verb, self.item_temp)
        self.TargetID = self.pair_temp.get_target_id(target_name)
        
    def get_feed_dict(self, uid, iid):
        uid = self.users.get_sample_index(uid)
        iid = self.items.get_sample_index(iid)
        feed_dict = {"user": uid, "item": iid, "task": self.task, "domain": self.domain}
        if self._return_profile:
            ufeat = self.users.get_features(uid)
            ifeat = self.items.get_features(iid)
            _, ids, masks, fids, segs = self.pair_temp.construct(**(ufeat | ifeat))
            feed_dict["pair_ids"] = np.array(ids)
            feed_dict["pair_masks"] = np.array(masks)
            feed_dict["pair_fids"] = np.array(fids)
            feed_dict["pair_segs"] = np.array(segs)
        if self._return_vector:
            feed_dict["user_feat"] = np.array(self.users.get_vector(uid))
            feed_dict["item_feat"] = np.array(self.items.get_vector(iid))
        return feed_dict

    def get_profiles(self, who="both"):
        lists = ["users", "items"] if who == "both" else [who]
        for _ in lists:
            prof = getattr(self, _)
            for idx in range(len(prof)):
                yield prof.get_profile(idx, False)

    def get_keywords(self, tokenize_=str.split, topK=None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from nltk.corpus import stopwords
        stopwords = stopwords.words("english") + ["none", 'might', 'must', 'need', 'ible', 'ical', 'ized']
        stopwords = set(stopwords + tokenize_(" " + " ".join(stopwords)))
        docs = []
        for idx in range(len(self.items)):
            item = self.items.get_features(idx)
            docs.append(" ".join("%s %s" % pair for pair in item.items() if pair[1]))
        tokenize = lambda s: [_ for _ in tokenize_(s) if len(_) > 2 and _[1:].isalpha() and _[1:].lower() not in stopwords]
        model = TfidfVectorizer(tokenizer=tokenize,
                                ngram_range=(1, 2),
                                sublinear_tf=True,
                                lowercase=False,
                                ).fit(docs)
        tfidfs = model.transform([' '.join(docs)]).todense().tolist()[0]
        unique, words = set(), []
        for pair in sorted(zip(tfidfs, model.get_feature_names_out()), reverse=True):
            for word in pair[1].split(" "):
                if word in unique:
                    continue
                unique.add(word)
                words.append(word)
                if len(words) == topK:
                    return words
        return words

    def list_features(self):
        return self.users.list_features() + self.items.list_features()

    def exhaustive_sampling(self, labels, keeps=None, drops=None):
        """we exhaustive both users and items profiels"""
        labels = [__ for _ in labels for __ in _]
        names = [s.name for s in self.pair_temp.slots]
        if keeps is not None:
            names = [_ for _ in names if _ in keeps]
        elif drops is not None:
            names = [_ for _ in names if _ not in drops]
        features = [f for f in self.list_features() if f["name"] in names]
        choices = [f["choices"] | {f["miss"]} for f in features]
        missing = [f["miss"] for f in features]
        names = [f['name'] for f in features]
        bar = tqdm.tqdm(total=functools.reduce(lambda x, y: x * y, map(len, choices)))
        for sample in itertools.product(*choices):
            sample = {n: v for n, v, m in zip(names, sample, missing)}
            ids = self.pair_temp.construct(**(sample | {"MASK_LABEL": labels[0]}))[1]
            text = self._tokenizer.decode(ids, skip_special_tokens=True)
            for label in labels:
                yield text.replace(labels[0], label)
            bar.update(1)

    def exhaustive_sampling(self, labels, fullsize=100000):
        """we only exhaustive user profiles"""
        labels = [__ for _ in labels for __ in _]
        fullnames = {s.name for s in self.pair_temp.slots}
        features = [f for f in self.users.list_features() if f["name"] in fullnames]
        choices = [f["choices"] | {f["miss"]} for f in features]
        missing = [f["miss"] for f in features]
        names = [f['name'] for f in features]
        user_size = functools.reduce(lambda x, y: x * y, map(len, choices))

        unique_items = {}
        for iid in range(len(self.items)):
            item_features = {k: v for k, v in self.items.get_features(iid).items() if k in fullnames} 
            tmp = (_[1] for _ in sorted(item_features.items(), key=lambda x: x[0]))
            if tmp not in unique_items:
                unique_items[tmp] = item_features | {"MASK_LABEL": labels[0]}
        
        bar = tqdm.tqdm(total=len(unique_items) * user_size)
        sampling = fullsize / len(labels) / len(unique_items) / user_size
        for item_features in unique_items.values():
            for user_features in itertools.product(*choices):
                bar.update(1)
                if random.random() <= sampling * 1.01:
                    try:
                        sample = {n: v for n, v, m in zip(names, user_features, missing) if v != m} | item_features
                        ids = self.pair_temp.construct(**sample)[1]
                        text = self._tokenizer.decode(ids, skip_special_tokens=True)
                        for label in labels:
                            yield text.replace(labels[0], label)
                    except AssertionError:
                        pass
                

class BaseData(tc.utils.data.Dataset):
    def __init__(self, subset, metaset, split, threshold, sampling):
        assert isinstance(split, str) and isinstance(threshold, float)
        assert isinstance(metaset, BaseMeta) and subset in ("full", "train", "test", "valid")
        super(tc.utils.data.Dataset).__init__()
        self.meta, self.users, self.items = metaset, metaset.users, metaset.items
        self.data = CorpusSearchIndex(metaset.root + r"/%s.tsv" % subset, sampling=sampling)
        self.seg, self.thres = split, threshold

    def __iter__(self):
        for row in self.data:
            record = row.split(self.seg)
            score = float(record[2])
            click = 1.0 if score >= self.thres else 0.0
            yield (self.users.get_sample_index(record[0]),
                   self.items.get_sample_index(record[1]),
                   score, click)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        uid, iid, rate = self.data[idx].split(self.seg)[:3]
        info = self.meta.get_feed_dict(uid, iid)
        info["score"] = float(rate)
        info["click"] = 1.0 if info["score"] >= self.thres else 0.0
        return info

    def get_max_fid(self):
        temp = self.meta.pair_temp
        return max(slot.fid for slot in temp.slots)

    def reset_meta(self, new_meta):
        assert new_meta.root == self.meta.root
        self.meta = new_meta
