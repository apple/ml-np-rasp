#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import numpy as np
from functools import partial
from tqdm.auto import trange

from add import *

VERBOSE = False

def pprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

def get_pad_len(x0, x1):
    max_digits = max(len(list(str(x0))), len(list(str(x1))))
    extra_padding = 1  # we pad by 1 to allow space for a carry in the output
    return max_digits + extra_padding

def format_prompt(x0, x1, pad_len):
    # x0, x1 = 6683, 345, ndigit = 4
    def num2array(num : int):
        str_digits = list(str(num).rjust(pad_len, '0'))
        digits = np.array([int(dig) for dig in str_digits])
        digits = np.array([[-100 - (i), d] for i, d in enumerate(digits)]).flatten()
        return digits
    return np.concatenate(([START_PROMPT], num2array(x0), [PLUS], num2array(x1), [EQUALS_SIGN]))


def output2result(out, reverse=False):
    idx_eq = list(out).index(EQUALS_SIGN)
    out = out[idx_eq+1:]
    out = out[out >= 0]  # strip out all special tokens, which are all negative
    if reverse:
        out = out[::-1]
    string = ''
    for digit in out:
        string += str(int(digit))
    return int(string)


def test_addition(x0, x1, func, reverse):
    pprint(f'{x0} + {x1} = {x0 + x1}')

    pad_len = get_pad_len(x0, x1)

    inp = format_prompt(x0, x1, pad_len=pad_len)

    out = func(inp)

    pprint('should get', x0 + x1)
    pprint('seq_out:', out)
    y = output2result(out, reverse=reverse)
    pprint('got', y)
    assert (y == x0 + x1)

def test_random_addition(ndigit, max_examples, func, reverse):
    for i in trange(max_examples):
        x0, x1 = np.random.randint(0, 10**ndigit, size=2)
        test_addition(x0, x1, func, reverse)


def ar_sample(prompt, next_tok_fn):
    pprint('prompt =', list(prompt))
    seq = prompt.copy()

    while seq[-1] != END_RESPONSE:
        next_tok = next_tok_fn(seq)[-1]
        # pprint('next token', next_tok)
        seq = np.concatenate((seq, [next_tok]))
    return seq



if __name__ == '__main__':
    ## test specific numbers:
    fwd_sample = partial(ar_sample, next_tok_fn=partial(next_tok_addition, reverse=False))
    rev_sample = partial(ar_sample, next_tok_fn=partial(next_tok_addition, reverse=True))
    test_addition(88, 842, func=fwd_sample, reverse=False)
    test_addition(88, 842, func=rev_sample, reverse=True)

    # Test random addition problems, both fwd and reverse order.
    for rev in [True, False]:
        sample_func = partial(ar_sample, next_tok_fn=partial(next_tok_addition, reverse=rev))
        test_random_addition(ndigit=5, max_examples=100, func=sample_func, reverse=rev)