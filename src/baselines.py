import sys
import os

import torch as tc

from datautils.movielens import MovieLensData, MovieLensMeta
from datautils.coupons import CouponsData, CouponsMeta
from datautils.restaurants import RestaurantMeta, RestaurantData

from metrics.ctr import CTREvaluator
from metrics.report import get_name

from models.utils import frozen
from models.randomly import RandomRecommend, PopularRecommend
from models.plms import PLM
from models.simples import PairwiseNSP, PairwiseContent, TargetName

SEED = int(sys.argv[1])
CUDA = str(sys.argv[2])
fpath = "../datasets/downstream_tasks/"
full_list = list(filter(lambda x: os.path.isdir(fpath + x), os.listdir(fpath)))
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
        train = cls[dataset][1](meta, "train")
        rater = CTREvaluator(cls[dataset][1](meta, "test"), batch_size=64)

        for architect in [RandomRecommend, PairwiseNSP, PairwiseContent, TargetName]:
            frozen(SEED)
            model = architect(meta=meta, dataset=train, plm=plmname, device=device)
            print("Dataset=%s" % dataset, "Subset=test", "Seed=%s" % SEED, "Model=%s" % get_name(model), rater.evaluate(model), sep="\t")
    
