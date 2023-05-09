import numpy as np
import pandas as pd



def get_name(model):
    arch = str(type(model))
    return arch.split(".")[-1][:-2]



class Report:
    def __init__(self, metrics=None, Ks=None):
        self._records = {} if metrics is None else {m: [] for m in metrics}
        self._k = [] if Ks is None else Ks
        self._records["_Summary"] = []

    def __len__(self):
        if len(self._records) == 0:
            return 0
        return len(self._records["_Summary"])

    def append(self, result, **kwrds):
        assert isinstance(result, dict)
        result.update(kwrds)
        for key in list(result):
            val = result[key]
            if isinstance(val, (list, tuple)):
                val = list(val)
                assert len(val) == len(self._k)
                for k, val in zip(self._k, val):
                    result[key + "@%s" % k] = val
                del result[key]
            
        current_keys = set(result)
        history_keys = set(self._records)
        size = len(self)
        for new_key in current_keys - history_keys:
            if not new_key.startswith("_"):
                self._records[new_key] = [None] * size
        for pad_key in history_keys - current_keys:
            result[pad_key] = None
        for key, val in result.items():
            self._records[key].append(val)          
        

    def summary(self, agg=np.mean, groupby=None, **where):
        if groupby is None:
            groupby = ["_Summary"]
        columns = sorted(self._records)
        table = np.array([self._records[c] for c in columns])
        for key, val in where.items():
            assert key in self._records, "Summary is denited: condition key `%s` is not an existing metric." % key
            table = table[:, table[columns.index(key)] == val]
        
        titles = ["Group"] + sorted(set(columns) - set(groupby))
        indexs = [columns.index(t) for t in titles[1:]]
        groups = {g: set(table[columns.index(g)]) for g in groupby}
            

        def get_subtable(table, groups):
            groups = groups.copy()
            key = sorted(groups)[0]
            vals = groups.pop(key)
            key_column = table[columns.index(key)]
            for val in vals:
                subtable = table[:, key_column == val]
                if len(subtable) == 0:
                    continue
                cond = "%s=%s" % (key, val)
                if len(groups) == 0:
                    row = [[cond]]
                    for i in indexs:
                        try:
                            row.append(agg(subtable[i]))
                        except Exception as e:
                            row.append("Error")
                    yield row
                    
                else:
                    for row in get_subtable(subtable, groups):
                        row[0].append(cond)
                        yield row

        reports = []
        for row in get_subtable(table, groups):
            row[0] = ",".join(row[0])
            reports.append(row)
                    
        table = pd.DataFrame(reports, columns=titles)
        if "_Summary" in table:
            del table["_Summary"]
        if groupby[0] == "_Summary":
            del table["Group"]
        return table

    def parse_logfile(self, fpath, word=None):
        print("Parsing logging file: %s" % fpath)
        if word is None:
            word = "\n" 
        with open(fpath) as logs:
            for row in logs:
                if word in row:
                    result = {}
                    for item in row.strip().split("\t"):
                        if "=" in item:
                            k, v = item.split("=", 1)
                            if k in result:
                                continue
                            for func in [float, eval]:
                                try:
                                    v = func(v.strip())
                                    break
                                except:
                                    pass 
                            result[k.strip()] = v
                        elif item.startswith("{") and item.endswith("}"):
                            result.update(eval(item.replace("nan", "float('nan')")))
                    self.append(result)

        

if __name__ == "__main__":
    report = Report(Ks=[1, 5, 10, 20, 50, 100])
    report.parse_logfile("../logs/baselines.log", "Test")
    print(report.summary(groupby=["Model", "Shot", "Test Dataset"]))


