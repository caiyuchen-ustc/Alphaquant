"""Robust reward + neighborhood robustness — the bug#3 fix.

AlphaGPT rewarded a formula by its TRAIN score, so RL searching thousands of formulas is
a p-hacking machine that surfaces train-lucky junk. Here the reward is the WORSE of train
and valid, minus a penalty on the train-valid gap — directly borrowing coin/bt/search.py's
`worse - penalty*gap` idea. The best formula is tracked on VALID. The TEST split never
enters the reward; it is scored exactly once at the very end (engine.finalize).
"""
from __future__ import annotations

from .backtest import quick_score
from .config import Cfg
from .vm import StackVM


def robust_reward(formula, feat_train, fwd_train, mask_train,
                  feat_valid, fwd_valid, mask_valid, vm: StackVM) -> float:
    """min(train, valid) - GAP_PENALTY * max(0, train - valid)."""
    s_train = vm.execute(formula, feat_train)
    if s_train is None:
        return -5.0
    q_train = quick_score(s_train, fwd_train, mask_train)
    if q_train <= -2.0:                       # degenerate on train, skip valid
        return q_train
    s_valid = vm.execute(formula, feat_valid)
    q_valid = quick_score(s_valid, fwd_valid, mask_valid)
    gap = max(0.0, q_train - q_valid)
    return min(q_train, q_valid) - Cfg.GAP_PENALTY * gap


def perturb(formula, feature_count, vocab_size, rng_int):
    """Return a formula variant: replace one token with another valid token."""
    if not formula:
        return list(formula)
    out = list(formula)
    pos = rng_int % len(out)
    # deterministic-ish swap driven by rng_int (avoids Math.random-style nondeterminism)
    new_tok = (out[pos] + 1 + (rng_int // len(out))) % vocab_size
    out[pos] = new_tok
    return out


def neighborhood_robustness(formula, feat, fwd, mask, vm: StackVM, n_neighbors=8) -> float:
    """Mean-minus-half-std of neighbor scores — prefer a plateau over a sharp peak.

    Used only in finalize() to re-rank top-K candidates, not in every RL step.
    """
    from .vocab import FORMULA_VOCAB
    scores = []
    base = vm.execute(formula, feat)
    scores.append(quick_score(base, fwd, mask))
    for i in range(1, n_neighbors + 1):
        var = perturb(formula, FORMULA_VOCAB.feature_count, FORMULA_VOCAB.size, i * 7 + 1)
        s = vm.execute(var, feat)
        scores.append(quick_score(s, fwd, mask))
    import numpy as np
    arr = np.array(scores)
    return float(arr.mean() - 0.5 * arr.std())
