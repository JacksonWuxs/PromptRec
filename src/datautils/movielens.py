import re
import collections


import torch as tc
import numpy as np
import transformers

from .core.dataset import BaseMeta, BaseData
from .core.templates import Template
from .core.verbalizers import (Continue, Category, Simple, Nominal, lower_spliter,
                               Zipcode, Occupation, MovieTitle)


class MovieLensMeta(BaseMeta):
    def __init__(self, root, tokenizer, prefix="", suffix="", task_id=0, domain_id=0, return_prof=True, return_vec=True, include_sep=False, include_mask=True):
        user_string = "The {:gender:} is a {:age:} {:occupation:} {:area:}. "
        item_string = "The {:title:} is categorized as a {:categories:} movie {:kid:} . "
        suffix = "The user is that {:gender:}, and the item is that movie. " + suffix
        slots = {"occupation": {"fid": 2, "prefix": ""},
                  "age": {"fid": 0},
                  "gender": {"fid": 1},
                  "categories": {"optional": True,},
                  "kid": {"optional": True,},
                "area": {"optional": True, "prefix": "living in "},
                }
        user_verbs = (Simple("UserID", "N/A"),
                       Continue("age", "nan", [13, 20, 35, 65], ["childhood", "teenage", "young", "middle aged", "old"]),
                       Category("gender", "N/A", {"M": "man", "F": "woman"}),
                       Occupation("occupation"),
                       Zipcode("area", "")
                       )
        item_verbs = (Simple("MovieID", "N/A"),
                      MovieTitle("title"),
                      Simple("release", "N/A"),
                      Nominal("categories", lower_spliter),
                      Category("kid", "N/A", {"1": "for children"}),
                      )
        BaseMeta.__init__(self, root, tokenizer, "\t", user_verbs, item_verbs, user_string, item_string, prefix, suffix, slots, 256, "title",
                          task_id, domain_id, return_prof, return_vec, include_sep)
        self.keeps = ['age', 'gender', 'occupation', 'title', 'categories']
        self.drops = None

    

class MovieLensData(BaseData):
    def __init__(self, metaset, subset, sampling=None):
        BaseData.__init__(self, subset, metaset, "\t", 4.0, sampling)



