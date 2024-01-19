#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import numpy as np
from .core import *

# np-rasp library functions

def equals(x, y):
    return x == y

def leq(x, y):
    return x <= y

def lt(x, y):
    return x < y

def geq(x, y):
    return x >= y

def gt(x, y):
    return x > y


def shift_right(x, n, default=0):
    # shifts sequence x to the right by n positions
    return kqv(indices(x) + n, indices(x), x, equals, default=default)

def cumsum(bool_array):
    # returns number of previous True elements in bool_array
    return sel_width(select(bool_array, bool_array, lambda k, q: k))

def where(condition, x_if, y_else):
    # equivalent to np.where(condition, x_if, y_else)
    x_masked = seq_map(x_if,    condition, lambda x, m: x if m else 0)
    y_masked = seq_map(y_else,  condition, lambda y, m: y if not m else 0)
    return seq_map(x_masked, y_masked, lambda x, y: x if y == 0 else y)

def mask(x, bool_mask, mask_val=0):
    # equivalent to x*bool_mask + default*(~bool_mask)
    return where(bool_mask, x, full(x, mask_val))


def maximum(x):
    return kqv(x, x, x, lambda k, q: True, reduction='max')

def minimum(x):
    return -maximum(-x)

def argmax(x):
    mm = maximum(x)
    return kqv(mm, x, indices(x), reduction='max')

def argmin(x):
    return argmax(-x)


def num_prev(x, queries):
    # output[i] = number of previous elements of x equal to queries[i], inclusive
    return sel_width(select(x, queries, equals))

def has_seen(x, queries):
    return kqv(x, queries, full(x, 1), equals, default=0)

def firsts(x, queries, default=-1):
    # find the index of the first occurrence of each query[i] in x
    # out[i] := np.flatnonzero(x[:i+1] == queries[i]).min() 
    return kqv(x, queries, indices(x), equals, default=default, reduction='min')

def lasts(x, queries, default=-1):
    # find the index of the last occurrence of each query[i] in x
    # out[i] := np.flatnonzero(x[:i+1] == queries[i]).max() 
    return kqv(x, queries, indices(x), equals, default=default, reduction='max')


def index_select(x, idx, default=0):
    # indexes into sequence x, via index sequence idx
    # i.e. return x[idx] if idx[i] <= i else default
    return kqv(indices(x), idx, x, equals, default=default)

def first_true(x, default=-1):
    # returns the index of the first true value in x
    seen_true = kqv(x, full(x, 1), full(x, 1), equals, default=0)
    first_occ = seq_map(seen_true, shift_right(seen_true, 1), lambda curr, prev : curr and not prev)
    return kqv(first_occ, full(x, 1), indices(x), equals, default=default)

def mode(x):
    num_prev_matching = sel_width(select(x, x, equals))
    idx = argmax(num_prev_matching)
    return index_select(x, idx)
    
def binary_mode(x):
    num_prev_zeros = sel_width(select(x, full(x, 0), equals))
    num_prev_ones  = sel_width(select(x, full(x, 1), equals))
    mode_val = seq_map(num_prev_ones, num_prev_zeros, gt)*1
    return mode_val

def induct_kqv(k, q, v, offset, default=0, null_val=-999):
    # get value of v at index of: first occurrence of q[i] found in k (if found) + offset.
    # (excludes the last OFFSET tokens of k from matching)
    # null_val is a special token that cannot appear in k or q; used to prevent accidental matches
    indices_to_copy = firsts(shift_right(k, offset, default=null_val), q, default=null_val)
    copied_values = index_select(v, indices_to_copy, default=default)
    return copied_values

def induct(k, q, offset, default=0, null_val=-999):
    return induct_kqv(k, q, k, offset=offset, default=default, null_val=null_val)

def induct_prev(k, q, offset, default=0, null_val=-999):
    # A version of induct for negative offsets.
    indices_to_copy = firsts(k, q, default=null_val) + offset
    copied_values = index_select(k, indices_to_copy, default=default)
    return copied_values