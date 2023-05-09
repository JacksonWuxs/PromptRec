import re
import collections


import torch as tc
import numpy as np
import transformers

from .core.dataset import BaseMeta, BaseData
from .core.templates import Template
from .core.verbalizers import Continue, Category, Simple, Functional


class CouponsMeta(BaseMeta):
    def __init__(self, root, tokenizer, prefix="", suffix="", task_id=0, domain_id=0, return_prof=True, return_vec=True, include_sep=False, include_mask=True):
        user_string = "It is a {:temprature:} {:weather:} {:time:}. The driver is a {:salary:} and {:age:} {:gender:} driving {:destination:} {:passanger:}. "
        item_string = "A {:coupon:} located in the {:direction_s:} {:direction_o:} driving direction sends a coupon that expires in {:expiration:}. "
        suffix = "The user is that driver, and the item is that coupon. " + suffix
        slots = {"age": {"fid": 0},
                      "gender": {"fid": 1},
                      "occupation": {"fid": 2},
                      "marry": {"fid": 3},
                      "salary": {"fid": 4},
                      "passanger": {"optional": True},
                 "direction_s": {"optional": True},
                      "direction_o": {"optional": True}}

        user_verbs = (Category("destination", "", {"No Urgent Place": "to somewhere", "Home": "home", "Work": "to the office"}),
                      Category("passanger", "", {"Friend(s)": "with friends", "Kid(s)": "with kids"}),
                      Category("weather", "", {"Rainy": "rainy", "Sunny": "sunny", "Snowy": "snowy"}),
                      Category("temprature", "", {"30": "freezing", "55": "cold", "80": "hot"}),
                      Category("time", "", {"7AM": "early morning", "10AM": "morning", "2PM": "afternoon", "6PM": "night", "10PM": "late night"}),
                      Category("gender", "", {"Male": "man", "Female": "woman"}),
                      Category("age", "", {"below21": "teenage", "21": "young", "26": "young", "31": "young", "36": "middle aged", "41": "middle aged", "46": "middle aged", "50plus": "old"}),
                      Category("marry", "", {"Married partner": "has married", "Unmarried partner": "is unmarried", "Single": "is still single", "Divorced": "is divorced", "Widowed": "is widowed"}),
                      Category("has_child", "", {"1": "a child", "0": "no child"}),
                      Category("degree", "", {"Associates degree": "bachelor degree", "High School Graduate": "no degree", "Some High School": "no degree", "Some college - no degree": "no degree", "Bachelors degree": "batchelor degree", "Graduate degree (Masters or Doctorate)": "graduate degree"}),
                      Functional("occupation", str.lower, ""),
                      Category("salary", "", {"Less than $12500": "low income", "$12500 - $24999": "low income", "$25000 - $37499": "low income", "$37500 - $49999": "low income",
                                              "$50000 - $62499": "moderate income", "$62500 - $74999": "moderate income", "$75000 - $87499": "moderate income",
                                              "$87500 - $99999": "high income", "$100000 or More": "high income"}))
        item_verbs = (Category("coupon", "", {"Restaurant(<20)": "famous restaurant",
                                              'Restaurant(20-50)': "restaurant",
                                              'Bar': "bar",
                                              'Carry out & Take away': "takeaway store",
                                              'Coffee House': "cafe"}),
                      Category("expiration", "", {"2h": "two hours", "1d": "one day"}),
                      Simple("car", ""),
                      Simple("parking"),
                      Simple("has_bar"),
                      Simple("has_cafe"),
                      Simple("top_20"),
                      Simple("top_50"),
                      Simple("unknow"),
                      Simple("unknow"),
                      Simple("unknow"),
                      Category("direction_s", "", {"1": "same"}),
                      Category("direction_o", "", {"1": "opposite"}),
                      )
        BaseMeta.__init__(self, root, tokenizer, "|", user_verbs, item_verbs, user_string, item_string, prefix, suffix, slots, 196, "coupon",
                          task_id, domain_id, return_prof, return_vec, include_sep)


    

class CouponsData(BaseData):
    def __init__(self, metaset, subset, sampling=None):
        BaseData.__init__(self, subset, metaset, "\t", 1.0, sampling)


