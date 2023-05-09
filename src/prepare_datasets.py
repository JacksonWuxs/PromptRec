import zipfile
import collections
import os
import random
import sys

from models.utils import frozen


SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42

def split_dataset(target, train_rate=250, valid_rate=50):
    frozen(SEED)
    with open(target + "/full.tsv", encoding="utf8") as f:
        for size, _ in enumerate(f, 1):
            pass
    if isinstance(train_rate, int):
        train_rate = train_rate / size
    if isinstance(valid_rate, int):
        valid_rate = valid_rate / size
    assert train_rate + valid_rate <= 1.
    recordlist = list(range(size))
    random.shuffle(recordlist)
    validset = set(recordlist[int(len(recordlist) * train_rate):int(len(recordlist) * (train_rate + valid_rate))])
    testset = set(recordlist[int(len(recordlist) * (train_rate + valid_rate)):])
    with open(target + "/full.tsv", encoding="utf8") as full,\
         open(target + "/train.tsv", "w", encoding="utf8") as train,\
         open(target + "/valid.tsv", "w", encoding="utf8") as valid,\
         open(target + "/test.tsv", "w", encoding="utf8") as test:
        #full.readline() # skip the column line
        for idx, row in enumerate(full):
            if idx in validset:
                valid.write(row)
            elif idx in testset:
                test.write(row)
            else:
                train.write(row)

def shuffle_dataset(data):
    frozen(SEED)
    with open(data, encoding="utf8") as f:
        records = f.readlines()
    random.shuffle(records)
    with open(data, 'w', encoding="utf8") as f:
        for row in records:
            f.write(row)

def clean_dataset_dir(target):
    valid = {"full.tsv", "train.tsv", "test.tsv", "valid.tsv",
              "item_idx.txt", "item_meta.txt", "user_idx.txt", "user_meta.txt"}
    for fpath in os.listdir(target):
        if fpath not in valid:
            os.remove(target + "/" + fpath)   
                

def prepare_movielen100k(source, target="./"):
    target = os.path.abspath(target).replace("\\", "/")
    with zipfile.ZipFile(source) as f:
        for old, new in zip(["u.user", "u.item", "ua.base", "ua.test"],
                        ["user_meta.txt", "item_meta.txt", "train.tsv", "test.tsv"]):
            f.extract("ml-100k/%s" % old, target)
            with open(target + "/ml-100k/%s" % old, encoding="ISO-8859-1") as fin,\
                 open(target + "/ml-100k/%s" % new, "w", encoding="utf8") as fout:
                for row in fin:
                    fout.write(row)
    for node in ["user", "item"]:
        with open(target + "/ml-100k/%s_meta.txt" % node, encoding="ISO-8859-1") as f,\
             open(target + "/ml-100k/%s_idx.txt" % node, "w") as t:
            for size, row in enumerate(f, 1):
                t.write(str(size) + "\n")

    with open(target + "/ml-100k/full.tsv", "w", encoding="ISO-8859-1") as fout:
        for subset in ["train", "test"]:
            with open(target + "/ml-100k/%s.tsv" % subset, encoding="utf8") as fin:
                for row in fin:
                    fout.write(row)
    split_dataset(target + "/ml-100k/")
    clean_dataset_dir(target + "/ml-100k")
    shuffle_dataset(target + "/ml-100k/train.tsv")


def prepare_restaurant(source, target="./"):
    target = os.path.abspath(target).replace("\\", "/")
    with zipfile.ZipFile(source) as f:
        for name in f.namelist():
            f.extract(name, target)

    def _merge_and_rewrite(root, part, fpaths):
        records = {}
        for fpath in fpaths:
            with open(root + fpath, encoding="ISO-8859-1") as f:
                keys = f.readline().strip().split(",")[1:]
                for row in f:
                    pid, vals = row.split(",", 1)
                    vals = vals.strip().split(",")
                    attrs = records.setdefault(pid, {})
                    for key, val in zip(keys, vals):
                        attrs.setdefault(key, []).append(val)
            os.remove(root + fpath)

        uniq_attrs = sorted({name for attrs in records.values() for name in attrs})
        with open(root + "/%s_idx.txt" % part, "w", encoding="utf8") as fidx,\
             open(root + "/%s_meta.txt" % part, "w", encoding="utf8") as fmeta:
            #fidx.write("EmptyLine\n")
            #fmeta.write("\t".join(uniq_attrs) + "\n")
            for rid, attrs in records.items():
                fidx.write("%s\n" % rid)
                fmeta.write("\t".join("|".join(attrs.get(name, ["None"])) for name in uniq_attrs) + "\n")
    _merge_and_rewrite(target, "item", ["/chefmoz%s.csv" % _ for _ in ["accepts", "cuisine", "hours4", "parking"]] + ["/geoplaces2.csv"])
    _merge_and_rewrite(target, "user", ["/user%s.csv" % _ for _ in ["cuisine", "payment", "profile"]])                 
    with open(target + "/rating_final.csv", encoding="ISO-8859-1") as src, \
         open(target + "/full.tsv", "w", encoding="utf8") as fout:
        src.readline() # skip the title line
        for row in src:
            fout.write(row)
    os.remove(target + "/rating_final.csv")
    split_dataset(target)
    clean_dataset_dir(target)
    shuffle_dataset(target + "/train.tsv")


def prepare_coupon(source, target="./", seed=2048, train_rate=0.7):
    target = os.path.abspath(target).replace("\\", "/")
    with zipfile.ZipFile(source) as f:
        f.extract("in-vehicle-coupon-recommendation.csv", target)
    users, items = {}, {}
    user_feats = {"destination", "passanger", "weather", "temperature", "time", "gender", "age", "maritalStatus", "has_children", "education", "occupation", "income"} 
    item_feats = {"coupon", "expiration", "car", "Bar", "CoffeeHouse", "CarryAway", "RestaurantLessThan20", "Restaurant20To50", "toCoupon_GEQ5min", "toCoupon_GEQ15min", "toCoupon_GEQ25min", "direction_same", "direction_opp"}
    with open(target + "/in-vehicle-coupon-recommendation.csv", encoding="ISO-8859-1") as src, \
         open(target + "/full.tsv", "w", encoding="utf8") as full, \
         open(target + "/user_meta.txt", "w", encoding="utf8") as umeta, \
         open(target + "/item_meta.txt", "w", encoding="utf8") as imeta, \
         open(target + "/user_idx.txt", "w", encoding="utf8") as uidx, \
         open(target + "/item_idx.txt", "w", encoding="utf8") as iidx:
        keys = src.readline().strip().split(",")
        for row in src:
            user, item, y = [], [], None
            values = row.strip().split(",")
            assert len(values) == len(keys)
            for key, val in zip(keys, values):
                if key in user_feats:
                    user.append(val)
                elif key in item_feats:
                    item.append(val)
                elif key == "Y":
                    y = val
            user, item = tuple(user), tuple(item)
            assert len(user) == len(user_feats)
            assert len(item) == len(item_feats)
            if user not in users:
                new_id = len(users)
                users[user] = new_id
                umeta.write("|".join(user) + "\n")
                uidx.write(str(new_id) + "\n")
            if item not in items:
                new_id = len(items)
                items[item] = new_id
                imeta.write("|".join(item) + "\n")
                iidx.write(str(new_id) + "\n")
                
            uid, iid = users[user], items[item]            
            full.write("%d\t%d\t%s\n" % (uid, iid, y))
    split_dataset(target)
    clean_dataset_dir(target)
    shuffle_dataset(target + "/train.tsv")
    
    
if __name__ == "__main__":
    prepare_movielen100k("../datasets/downstream_tasks/ml-100k.zip",
                         "../datasets/downstream_tasks/")
    prepare_restaurant("../datasets/downstream_tasks/RCdata.zip",
                         "../datasets/downstream_tasks/mexico_restaurant")
    prepare_coupon("../datasets/downstream_tasks/in-vehicle-coupon-recommendation.csv.zip",
                         "../datasets/downstream_tasks/coupon")

