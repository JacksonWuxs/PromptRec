import math
import sys
import os
import copy
import pickle
import random


SEED = int(sys.argv[1])
CUDA = str(sys.argv[2])


import numpy as np
import torch as tc

from datautils.movielens import MovieLensData, MovieLensMeta
from datautils.coupons import CouponsData, CouponsMeta
from datautils.restaurants import RestaurantMeta, RestaurantData
from datautils.downstreams import CombineData

from metrics.ctr import CTREvaluator
from metrics.rank import RankEvaluator
from metrics.report import get_name

from models.utils import frozen
from models.randomly import RandomRecommend, PopularRecommend
from models.plms import PLM
from models.NeuMF import NeuMF
from models.Pnn import Pnn
from models.WideDeep import WideDeep
from models.LightGCN import LightGCN
from models.simples import PairwiseNSP, PairwiseContent, TargetName


fpath = "../datasets/downstream_tasks/"
full_list = list(filter(lambda x: os.path.isdir(fpath + x), os.listdir(fpath)))
shot = int(sys.argv[3])
plmname = "bert-large-uncased"
device = "cuda:%s" % CUDA
tc.cuda.set_device(int(CUDA))
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA
os.environ["TOKENIZERS_PARALLELISM"] = "true"
cls = {"mexico_restaurant": (RestaurantMeta, RestaurantData),
       "ml-100k": (MovieLensMeta, MovieLensData),
       "coupon": (CouponsMeta, CouponsData)}



if __name__ == "__main__":
    prefix, suffix = '', ''
    tokenizer = PLM.load_tokenizer(plmname)
    for dataset in full_list:
        frozen(SEED)
        meta = cls[dataset][0](fpath + dataset, tokenizer, prefix, suffix, domain_id=0)
        sample_train = cls[dataset][1](meta, "train", sampling=shot)
        valid = cls[dataset][1](meta, "valid")
        test = cls[dataset][1](meta, "test")
        rater = CTREvaluator(test, batch_size=64)
        
        if shot == 0:
            for architect in [RandomRecommend, PairwiseNSP, PairwiseContent, TargetName]:
                frozen(SEED)
                model = architect(meta=meta, dataset=sample_train, plm=plmname, device=device)
                print("Dataset=%s" % dataset, "Subset=test", "Seed=%s" % SEED, "Shot=0", "Model=%s" % get_name(model), rater.evaluate(model), sep="\t")
        else:
            for architect in [PopularRecommend,]:
                frozen(SEED)
                model = architect(meta=meta, dataset=sample_train, dropout=0.2, plm=plmname, device=device)
                model.fit(sample_train, valid=valid, lr=1e-3, bs=8, epochs=50)
                print("Dataset=%s" % dataset, "Subset=test", "Seed=%s" % SEED, "Shot=%d" % shot, "Model=%s" % get_name(model), rater.evaluate(model), sep="\t")


