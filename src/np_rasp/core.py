#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import numpy as np

## np-rasp core

def full(x, const):
    return np.full_like(x, const, dtype=int)

# fill = full
    
def indices(x):
    return np.arange(len(x), dtype=int)

def tok_map(x, func):
    return np.array([func(xi) for xi in x]).astype(int)

def seq_map(x , y, func):
    return np.array([func(xi, yi) for xi, yi in zip(x,y)]).astype(int)


def select(k, q, pred, causal=True):
    s = len(k)
    A = np.zeros((s, s), dtype=bool)
    for qi in range(s):
        for kj in (range(qi+1) if causal else range(s)):  # k_index <= q_index if causal
            A[qi, kj] = pred(k[kj], q[qi])
    return A

def sel_width(A):
    return np.dot(A, np.ones(len(A))).astype(int)

def aggr_mean(A, v, default=0):
    out = np.dot(A, v)
    norm = sel_width(A)
    out = np.divide(out, norm, out=np.full_like(v, default,dtype=float), where=(norm != 0))
    return out.astype(int)

def aggr_max(A, v, default=0):
    out = np.full_like(v, default)
    for i, row in enumerate(A):
        idxs = np.flatnonzero(row)
        if len(idxs) > 0:
            out[i] = np.max(v[idxs]) # max of selected elements in v
    return out.astype(int)

def aggr(A, v, default=0, reduction='mean'):
    if reduction == 'mean':
        return aggr_mean(A, v, default)
    elif reduction == 'max':
        return aggr_max(A, v, default)
    elif reduction == 'min':
        return -aggr_max(A, -v, -default)


def kqv(k, q, v, pred, default=0, reduction='mean'):
    return aggr(select(k, q, pred), v, default=default, reduction=reduction)

