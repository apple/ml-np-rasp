#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from np_rasp import *


START_PROMPT = -1
PLUS = -2
EQUALS_SIGN = -3
END_RESPONSE = -5
NONE = -88


def mask_between_tokens(seq, tok0, tok1):
    seen_tok0 = has_seen(seq, full(seq, tok0))
    seen_tok1 = has_seen(seq, full(seq, tok1))  
    ind_between = seq_map(seen_tok0, seen_tok1, lambda a, b: a and not b)  # ind(tok0) <= (*) < ind(tok1)
    return ind_between

def _add_safe(x, y):
    return x + y if (x >= 0) else x # preserve index-hints

def next_tok_addition(seq, reverse=True):
    prompt_mask = 1-has_seen(seq, full(seq, EQUALS_SIGN))
    second_summand_mask = mask_between_tokens(seq, PLUS, EQUALS_SIGN)
    prompt = mask(seq, prompt_mask)
    
    # let's first align the 1st summand with the second.
    other_summand_digit = induct(k=prompt, q=shift_right(prompt, 1), offset=1)
    pairsums = seq_map(seq, other_summand_digit, _add_safe)  # this puts pairsums aligned with the 2nd summand of the prompt
    pairsums = mask(pairsums, second_summand_mask, NONE)
    pairsums_nh = mask(pairsums, (seq >= 0), NONE) # nohints: only keep digits
    
    curr_output_digit  = shift_right(seq, 1)
    curr_pairsum = induct(pairsums, shift_right(seq, 2), offset=1) # pairsum that generated curr_output_digit
    next_pairsum = induct(pairsums, seq, offset=1)

    ## START CHANGES (FWD vs. REV order)
    if reverse:
        direct_carry = curr_pairsum > 9  # previous sum gives carry
        indirect_carry = (curr_pairsum == 9) & (curr_output_digit == 0)  # previous sum is 9 and earlier sum gave carr
        next_tok_gets_carry = direct_carry | indirect_carry

        # (simple) index-hint computations:
        final_hint = full(seq, -100) # final hint output is always -100 
        first_hint =  induct_prev(seq, full(seq, EQUALS_SIGN), offset=-2) # first hint is 2 places before '=' 
        next_hint = shift_right(seq, 1) + 1 
        eos = (next_hint > final_hint)
    else:
        gives_carry = tok_map(pairsums_nh, lambda _x: 1 if _x > 9 else 0)
        z = cumsum((pairsums_nh != 9) & (pairsums_nh != NONE))
        u = mask(z, gives_carry, mask_val=NONE)
        v = tok_map(u, lambda _x: _x - 1)
        chain_end_idxs = firsts(z, v, default=NONE)   # (left) ending indices of carry-chain
        
        curr_tok_got_carry = ((curr_pairsum % 10) != curr_output_digit)
        next_tok_inside_carry_chain =  (next_pairsum == 9) & curr_tok_got_carry 
            # in the middle of a carry-chain? (NOTE: assumes the pairsums has first element 0)
        
        next_tok_idx = kqv(pairsums, seq, indices(seq), equals) + 1
            # which answer-position are we at? (indices aligned to pairsums)
        next_tok_chain_end = kqv( chain_end_idxs , next_tok_idx , full(seq, 1), equals, default=0)
            # does the next_tok get a carry from the end of a carry-chain?
        next_tok_gets_carry = next_tok_inside_carry_chain | next_tok_chain_end

        # (simple) index-hint computations:
        final_hint = induct_prev(seq, full(seq, EQUALS_SIGN), offset=-2) # final hint is 2 places before '='
        first_hint = full(seq, -100)  
        next_hint = shift_right(seq, 1) - 1 
        eos = (next_hint < final_hint)
    ## END CHANGES

    next_tok = next_pairsum
    next_tok += next_tok_gets_carry
    next_tok = next_tok % 10

    ## Finally, handle the case of outputting index-hints
    next_tok_is_index_hint = (seq > -100) # all index-hints are <= -100
    eos = (eos & next_tok_is_index_hint)

    next_tok = where( next_tok_is_index_hint, next_hint, next_tok)
    next_tok = where( eos, full(seq, END_RESPONSE), next_tok)
    next_tok = where( (seq == EQUALS_SIGN), first_hint, next_tok) 
    return next_tok

