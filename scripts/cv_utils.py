"""
Small helpers for cross-validation (numpy only, so they can be tested without PyTorch).
"""
import numpy as np


def stratified_folds(y, k, seed):
    """Split clip indices into k folds so every fold has about the same share of each word.
    Returns a list of k index arrays; together they cover every clip exactly once."""
    rng = np.random.default_rng(seed)
    # deal each word's shuffled clips round-robin, starting at a random fold
    folds = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        start = int(rng.integers(k))
        for j, i in enumerate(idx):
            folds[(start + j) % k].append(int(i))
    return [np.array(sorted(f)) for f in folds]


def stratified_split(y, idx, frac, seed):
    """Take about `frac` of the clips in idx (at least one per word) for validation.
    Returns (train_idx, val_idx)."""
    rng = np.random.default_rng(seed)
    val = []
    for c in np.unique(y[idx]):
        ci = idx[y[idx] == c].copy()
        rng.shuffle(ci)
        n = max(1, int(round(frac * len(ci))))
        val.extend(ci[:n].tolist())
    val = np.array(sorted(val))
    train = np.array(sorted(set(idx.tolist()) - set(val.tolist())))
    return train, val


def wilson(correct, n, z=1.96):
    """95% confidence interval for an accuracy of `correct` out of `n` clips."""
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def confusion_pairs(y, pred, names, top=10):
    """Most frequent mistakes as (true word, predicted word, count)."""
    pairs = {}
    for t, p in zip(y, pred):
        if t != p:
            pairs[(int(t), int(p))] = pairs.get((int(t), int(p)), 0) + 1
    out = sorted(pairs.items(), key=lambda kv: -kv[1])[:top]
    return [(names[t], names[p], n) for (t, p), n in out]
