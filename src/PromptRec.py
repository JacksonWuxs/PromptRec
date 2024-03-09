import sys
import os
import time

import numpy as np
import torch as tc

from datautils.movielens import MovieLensData, MovieLensMeta
from datautils.coupons import CouponsData, CouponsMeta
from datautils.restaurants import RestaurantMeta, RestaurantData
from datautils.downstreams import CombineData

from models.utils import frozen
from metrics.ctr import CTREvaluator
from models.plms import PLM
from models.ours import TransferSoftPromptRecommander


SEED = int(sys.argv[1])
CUDA = str(sys.argv[2])
device = "cuda:%s" % CUDA
tc.cuda.set_device(int(CUDA))
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA


shots = int(sys.argv[3]) if len(sys.argv) > 3 else None
fpath = "../datasets/downstream_tasks/"
full_list = list(filter(lambda x: os.path.isdir(fpath + x), os.listdir(fpath)))
cls = {"mexico_restaurant": (RestaurantMeta, RestaurantData),
       "ml-100k": (MovieLensMeta, MovieLensData),
       "coupon": (CouponsMeta, CouponsData)}
plms = [
        (True, "../../IIFT/weights/llama/hf_models/7B/", "clm", "", "In short, the user's attitude towards the item is [::MASK::]"),
        (True, "prajjwal1/bert-tiny", "mlm", "", "In short, the user feels [::MASK::] about the item."),
        (True, "prajjwal1/bert-mini", "mlm", "", "In short, the user feels [::MASK::] about the item."),
        (True, "prajjwal1/bert-small", "mlm", "", "In short, the user feels [::MASK::] about the item."),
        (True, "bert-large-uncased", "mlm", "", "In short, the user feels [::MASK::] about the item."),
        (False, "t5-large", "s2s", "sst2 sentence:", "[::MASK::]"),
        ]

if __name__ == "__main__":
    target_labels = [["negative", ], ["positive", ]]
    for mixprec, plmname, headname, prefix, suffix in plms:
        frozen(SEED)
        tokenizer = PLM.load_tokenizer(plmname)
        plm = PLM(plmname, headname, tokenizer, device=device)
        for dataset in full_list:            
            frozen(SEED)
            target_meta = cls[dataset][0](fpath + dataset, tokenizer, prefix, suffix, domain_id=0)
            target_test = CTREvaluator(cls[dataset][1](target_meta, "test"), batch_size=32)

            frozen(SEED)
            model = TransferSoftPromptRecommander(plm, tasks=(1, 10, None),  domains=(1, 10, None), labels=target_labels, device=device)
            frozen(SEED)
            print("Dataset=%s" % dataset, "Seed=%s" % SEED, "Model=%s" % plmname, "Shot=0", target_test.evaluate(model, mixprec), sep="\t")

            del model
            tc.cuda.empty_cache()
            time.sleep(5)




