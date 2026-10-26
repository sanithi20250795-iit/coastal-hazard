"""Time-blocked cross-validation: contiguous blocks with a gap so the model never trains on the future."""
import numpy as np


def time_folds(n, k=5, gap=48):
    edges = np.linspace(0, n, k + 1).astype(int)
    for i in range(k):
        lo, hi = edges[i], edges[i + 1]
        test = np.arange(lo, hi)
        train = np.concatenate([np.arange(0, max(lo - gap, 0)), np.arange(min(hi + gap, n), n)])
        if len(train) and len(test):
            yield train, test
