import re
import collections


import torch as tc
import numpy as np
import transformers

from .core.dataset import BaseMeta, BaseData
from .core.templates import Template
from .core.verbalizers import (Continue, Category, Simple,
                               Zipcode, Occupation, MovieTitle)


class MovieLensMeta(BaseMeta):
    def __init__(self, root, tokenizer, prefix="", suffix="", task_id=0, domain_id=0, return_prof=True, return_vec=True, include_sep=False, include_mask=True):
        user_string = "The {:gender:} is a {:age:} {:occupation:} {:area:}. "
        item_string = "The {:title:} is categorized as a" +\
                      "{:action:} {:adventure:} {:animation:} {:comedy:} {:documentary:} {:drama:} {:fantasy:} {:dark:} {:horror:} {:musical:} " +\
                      "{:mystery:} {:romantic:} {:scientific:} {:thriller:} {:western:} {:war:} movie {:kid:} . "
        suffix = "The user is that {:gender:}, and the item is that movie. " + suffix
        slots = {"occupation": {"fid": 2, "prefix": ""},
                      "age": {"fid": 0},
                      "gender": {"fid": 1},
                 "action": {"optional": True,},
                        "adventure": {"optional": True,},
                        "animation": {"optional": True,},
                        "kid": {"optional": True,},
                        "comedy": {"optional": True,},
                        "documentary": {"optional": True,},
                        "drama": {"optional": True,},
                        "fantasy": {"optional": True,},
                        "dark": {"optional": True,},
                        "horror": {"optional": True, },
                        "musical": {"optional": True,},
                        "mystery": {"optional": True,},
                        "romantic": {"optional": True,},
                        "scientific": {"optional": True,},
                        "thriller": {"optional": True,},
                        "war": {"optional": True,},
                        "western": {"optional": True,},
                        "area": {"optional": True, "prefix": "living in"},
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
                      Simple("unknown", "N/A"),
                      Simple("URL", "N/A"),
                      Simple("unknown", "N/A"),
                      Category("action", "N/A", {"1": "action"}),
                      Category("adventure", "N/A", {"1": "adventure"}),
                      Category("animation", "N/A", {"1": "animated"}),
                      Category("kid", "N/A", {"1": "for children"}),
                      Category("comedy", "N/A", {"1": "comedy"}),
                      Category("crime", "N/A", {"1": "criminal"}),
                      Category("documentary", "N/A", {"1": "documentary"}),
                      Category("drama", "N/A", {"1": "dramatic"}),
                      Category("fantasy", "N/A", {"1": "fantasy"}),
                      Category("dark", "N/A", {"1": "dark"}),
                      Category("horror", "N/A", {"1": "horrible"}),
                      Category("musical", "N/A", {"1": "musical"}),
                      Category("mystery", "N/A", {"1": "mystical"}),
                      Category("romantic", "N/A", {"1": "romantic"}),
                      Category("scientific", "N/A", {"1": "scientific"}),
                      Category("thriller", "N/A", {"1": "horrible"}),
                      Category("war", "N/A", {"1": "war"}),
                      Category("western", "N/A", {"1": "western"}),
                      )
        BaseMeta.__init__(self, root, tokenizer, "|", user_verbs, item_verbs, user_string, item_string, prefix, suffix, slots, 196, "title",
                          task_id, domain_id, return_prof, return_vec, include_sep)


    

class MovieLensData(BaseData):
    def __init__(self, metaset, subset, sampling=None):
        BaseData.__init__(self, subset, metaset, "\t", 4.0, sampling)



