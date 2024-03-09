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
cls = {"mexico_restaurant": (RestaurantMeta, RestaurantData),
       "ml-100k": (MovieLensMeta, MovieLensData),
       "coupon": (CouponsMeta, CouponsData)}
plms = [
        ("prajjwal1/bert-tiny", "mlm", "", "In short, the user feels [::MASK::] about the item."),
        ("prajjwal1/bert-mini", "mlm", "", "In short, the user feels [::MASK::] about the item."),
        #("prajjwal1/bert-small", "mlm", "", "In short, the user feels [::MASK::] about the item."),
        ]


if __name__ == "__main__":
    ntask = nhints = 10
    target_labels = [["negative"], ["positive"]]
    for plmname, headname, prefix, suffix in plms:
        frozen(SEED)
        tokenizer = PLM.load_tokenizer(plmname)
        plm = PLM(plmname, headname, tokenizer, device=device)
        for dataset in full_list:
            target_meta = cls[dataset][0](fpath + dataset, tokenizer, prefix, suffix, domain_id=0)
            target_valid = CTREvaluator(cls[dataset][1](target_meta, "valid"), batch_size=128)
            target_test = CTREvaluator(cls[dataset][1](target_meta, "test"), batch_size=128)


            first_round = True
            for nhints in [5, 10, 50, 100, 200]:
                for ntask in [10, 50,]:
                    hints = [target_meta.get_keywords(tokenizer.tokenize, nhints)] 
                    sources = []
                    for source, (meta_cls, data_cls) in cls.items():
                        if source != dataset:
                            source_meta = meta_cls(fpath + source, tokenizer, 
                                           prefix, suffix, return_vec=False,
                                           domain_id=len(sources)+1)
                            hints.append(source_meta.get_keywords(tokenizer.tokenize, nhints))
                            sources.append(data_cls(source_meta, "full"))
                    sources = CombineData(sources)

                    frozen(SEED)
                    model = TransferSoftPromptRecommander(plm, tasks=(1, ntask, None), 
                                                  domains=(len(full_list), nhints, hints), 
                                                  dropout=0.0, labels=target_labels, device=device)
                    if first_round:
                        print("Dataset=%s" % dataset, "Seed=%s" % SEED, "Model=%s" % plmname, "Shot=0", target_test.evaluate(model, True), sep="\t")
                        first_round = False
            
                    model.fit(data=sources, bs=64, lr=1e-4, steps=50000, weight_decay=0., valid=target_valid, mode="pretrain")
                    print("Subset=Valid", "Tasks=%d" % ntask, "Hints=%d" % nhints, "Dataset=%s" % dataset, 
                            "Seed=%s" % SEED, "Model=%s" % plmname, "Shot=Transfer", target_valid.evaluate(model, True), sep="\t")
                    print("Subset=Test", "Tasks=%d" % ntask, "Hints=%d" % nhints, "Dataset=%s" % dataset, 
                            "Seed=%s" % SEED, "Model=%s" % plmname, "Shot=Transfer", target_test.evaluate(model, True), sep="\t")
                    del model
                    tc.cuda.empty_cache()
                    time.sleep(10)




