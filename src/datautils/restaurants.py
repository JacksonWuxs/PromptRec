import re
import collections

import torch as tc
import numpy as np
import transformers

from .core.dataset import BaseMeta, BaseData
from .core.templates import Template
from .core.verbalizers import Nominal, Continue, Category, Simple, Zipcode, lower_spliter


class RestaurantMeta(BaseMeta):
    def __init__(self, root, tokenizer, prefix='', suffix='', task_id=0, domain_id=0, return_prof=True, return_vec=True, include_sep=False, include_mask=True):
        user_string = "Our customer is a {:marry:} {:age:} woman. " +\
                      "She is looking for {:user_cuisine:} food {:budget:}. " +\
                      "{:user_dress:} {:user_payment:} " +\
                      "{:interest:} " 
        item_string = "{:name:} is a {:price:} restaurant selling {:res_cuisine:} food. " +\
                      "{:res_payment:} {:service:} {:res_dress:} "
        suffix = "The user is that woman, and the item is that restaurant. " + suffix
        slots = {"user_ambience": {"optional": True, "prefix": "for "},
                    "user_cuisine": {'maxlen': 10},
                    "user_dress": {"optional": True, "prefix": "The woman prefers a ", "suffix": " attire restaurant."},
                    "user_payment": {"optional": True, "prefix": "She will pay the restaurant in ", "suffix": "."},
                    "marry": {"optional": True},
                    "transport": {"optional": True, "prefix": "She will ", "suffix": " to the restaurant."},
                    "interest": {"optional": True, "prefix": "She also expects the restaurant can provide ", "suffix": "."},
                    "res_ambience": {"optional": True, "prefix": "shows a ", "suffix": " dining atmosphere and "},
                    "res_payment": {"optional": True, "prefix": "The restaurant supports payments ", "suffix": "."},
                    "res_accessibility": {"optional": True, "suffix": "disabled facilities,"},
                    "res_parking": {"optional": True, "suffix": "parkings,"},
                    "price": {"optional": True, "prefix": "", "suffix": ""},
                    "res_dress": {"optional": True, "prefix": "The restaurant requires customers to wear ", "suffix": " attire."},
                    "service": {"optional": True, "prefix": "The restaurant also provides ", "suffix": "."},
                    "age": {"fid": 0},
                    "budget": {"fid": 4, "optional": True, "prefix": "with a ", "suffix": " budget"},
                    }
        user_verbs = (Nominal("user_cuisine", lower_spliter), #
                      Nominal("user_payment", lower_spliter), #
                      Simple("activity", "?"),
                      Category("user_ambience", "?", {"friends": "her friends", "family": "her family", "solitary": "herself"}), #
                      Continue("age", "?", [1960, 1985], ["old", "middle aged", "young"]), #
                      Simple("budget", "?"), #
                      Simple("color", "?"),
                      Category("user_dress", "?", {"formal": "formal", "elegant": "elegant", "informal": "informal"}), #
                      Simple("drink", "None"),
                      Simple("height", "None"),
                      Simple("hijos", "?"),
                      Simple("interest", "none"),
                      Simple("latitude"),
                      Simple("longitude"),
                      Category("marry", "?", {"married": "married", "single": "single", "widow": "widowed"}), #
                      Simple("personality"), #
                      Simple("religion", "none"),
                      Simple("smoker", "?"),
                      Category("transport", "?", {"public": "take a bus", "on foot": "walk", "car owner": "drive"}),
                      Simple("weight")
                      )
        item_verbs = (Simple("res_ambience", "None"), #
                      Nominal("res_cuisine", lower_spliter), #
                      Nominal("res_payment", lower_spliter), #
                      Category("res_accessibility", "None", {"completely": "complete", "partially": "partial"}),
                      Simple("address"),  
                      Nominal("has_bar", lower_spliter, "None"),
                      Simple("area"),
                      Simple("city"), #
                      Simple("country"),
                      Simple("days"),
                      Simple("res_dress", "None"),
                      Simple("fax"),
                      Category("franchise", "None", {"t": "true", "f": "false"}),
                      Simple("hours"),
                      Simple("latitude"),
                      Simple("longitude"),
                      Simple("name"), #
                      Category("service", "None", {"Internet": "free internet", "variety": "variety"}), #
                      Category("res_parking", "None", {"fee": "paid", "valet parking": "valet", "validated parking": "paid", "yes": "free", "none": "paid", "public": "public", "street": "street"}), #
                      Category("price", "none", {"high": "expensive", "medium": "cheap", "low": "cheap"}),
                      Simple("res_smoking", "None"),
                      Simple("state"),
                      Simple("geometer"),
                      Simple("URL"),
                      Zipcode("area", "None")
                      )
        BaseMeta.__init__(self, root, tokenizer, "\t", user_verbs, item_verbs, user_string, item_string, prefix, suffix, slots, 256, "name",
                                                    task_id, domain_id, return_prof, return_vec, include_sep)
        self.keeps = None
        self.drops = ['URL', 'state', 'address', 'latitude', 'longitude', 'name']

class RestaurantData(BaseData):
    def __init__(self, metaset, subset, sampling=None):
        BaseData.__init__(self, subset, metaset, ",", 2.0, sampling)


