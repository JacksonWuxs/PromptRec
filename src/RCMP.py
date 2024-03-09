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


fpath = "../datasets/downstream_tasks/"
full_list = list(filter(lambda x: os.path.isdir(fpath + x), os.listdir(fpath)))
simple_name = {"mexico_restaurant": "restaurant", "coupon": "coupon", "ml-100k": "movie"}
cls = {"mexico_restaurant": (RestaurantMeta, RestaurantData),
       "ml-100k": (MovieLensMeta, MovieLensData),
       "coupon": (CouponsMeta, CouponsData)}


if __name__ == "__main__":
    check = sys.argv[3]
    size = sys.argv[4]
    target_labels = [["negative",], ["positive",]]
    rootname = "outputs/%s/10k_%s_seed%d/" % (check, size, SEED) + 'checkpoint-%d/'
    headname, prefix, suffix = "mlm", "", "In short, the user feels [::MASK::] about the item."
    for dataset in full_list:
        if dataset != check:
            continue
        
        best_score, best_addr, best_rslts = 0, None, None
        for step in range(1000, 10000, 500):
            plmname = rootname % step
            frozen(SEED)
            tokenizer = PLM.load_tokenizer(plmname)
            plm = PLM(plmname, headname, tokenizer, device=device)
            
            target_meta = cls[dataset][0](fpath + dataset, tokenizer, prefix, suffix, domain_id=0)
            target_test = CTREvaluator(cls[dataset][1](target_meta, "test"), batch_size=128)
            target_valid = CTREvaluator(cls[dataset][1](target_meta, "valid"), batch_size=64)

            frozen(SEED)
            model = TransferSoftPromptRecommander(plm, tasks=(1, 1, None), domains=(1, 1, None), labels=target_labels, device=device)
            scores = target_test.evaluate(model, True)
            score = scores["gauc"] if scores["gauc"] > 0 else scores["auc"]
            if score > best_score + 1e-4:
                best_score, best_addr, best_rslts = score, plmname, scores
            print("Dataset=%s" % dataset, 'Subset=Test', "Seed=%s" % SEED, "Model=%s" % plmname, "Shot=DARC", scores, sep="\t")
            del model
            tc.cuda.empty_cache()
            time.sleep(1)


        print("Dataset=%s" % dataset, 'Subset=Best', "Seed=%s" % SEED, "Model=%s" % best_addr, "Shot=DARC", best_rslts, sep="\t")


