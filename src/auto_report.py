import sys
import os

import numpy as np

from metrics.report import Report


if __name__ == "__main__":
    src = sys.argv[1]
    report = Report(Ks=[1, 5, 10, 20, 50, 100])
    report.parse_logfile(src, "gauc")
    report.summary(agg=np.mean, groupby=["Model", "Dataset"]).to_csv(src.replace(".log", "_mean.tsv"), sep="\t")
    report.summary(agg=np.std, groupby=["Model", "Dataset"]).to_csv(src.replace(".log", "_std.tsv"), sep="\t")
