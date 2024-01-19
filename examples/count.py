#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import numpy as np
from np_rasp import *

START_TOK = -1
END_TOK = -2

VERBOSE = False
def pprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

def count(x):
    # First, find the index of most-recent START_TOK (begninng of current xuence)
    start_idx =  kqv(x, full(x, START_TOK), indices(x), equals, reduction='max')
    pprint(start_idx, 'start_idx')
    
    # Then, fetch the start/end numbers of the current xuence
    start_nums = kqv(indices(x), start_idx+1, x, equals)
    end_nums   = kqv(indices(x), start_idx+2, x, equals)
    pprint(start_nums, 'start_nums')
    pprint(end_nums, 'end_nums')

    # Bool arrays:  whether we're predicting the first / last tokens of the current xuence
    is_first_pos = seq_map(indices(x), start_idx+2, equals)
    is_highest_num = seq_map(x, end_nums, equals)
    pprint(is_first_pos, 'is_first_pos')
    pprint(is_highest_num, 'is_highest_num')
    
    incr = tok_map(x, lambda i: i+1)  # incr := (x+1)
    pprint(incr, 'incr')
    
    incr_with_eos = where(is_highest_num, full(x, END_TOK), incr)
    pprint(incr_with_eos, 'next_tok')
    
    next_tok = where(is_first_pos, start_nums, incr_with_eos)
    pprint(next_tok, 'next_tok')
    return next_tok

## A simplified version of the above program, using library functions.
def count_lib(x):
    start_idx = lasts(x, full(x, START_TOK))
    
    # Then, compute the start/end numbers of the current sequence
    start_nums = index_select(x, start_idx+1)
    end_nums   = index_select(x, start_idx+2)

    # Bool arrays: whether we're predicting the first / last tokens of the current sequence
    pred_first_pos = seq_map(indices(x), start_idx+2, equals)
    pred_final_pos = (~pred_first_pos) & seq_map(x, end_nums, equals) # shorthand for tok_maps

    next_tok = where(pred_first_pos,             # if predicting the first token:
                     start_nums,                 #    next_tok = starting num
                     where(pred_final_pos,       # else if predicting the final token:
                           full(x, END_TOK),     #    next_tok = END_TOK                
                           x + 1))               # else: next_tok = prev_tok + 1
    return next_tok


if __name__ == '__main__':
    ## Test it on an input:
    input = np.array([5,6, START_TOK, 0, 4, 0,1,2,3,4,END_TOK, START_TOK, 5, 8,5,6,7,8])

    output = count(input)
    print('inp:\t', input)
    print('out:\t', output)


    ## Generate auto-regressively
    prompt = np.array([START_TOK, 0, 15])
    print('prompt:', prompt)
    x = prompt.copy()
    while x[-1] != END_TOK:
        next_tok = count(x)[-1]
        print(f'{x} -> {next_tok}')
        x = np.concatenate((x, [next_tok]))

    print('generation:', x)