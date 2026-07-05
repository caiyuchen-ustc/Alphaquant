"""StackVM: execute a postfix token formula into a [T, N] signal tensor.

feat_tensor is [N_feat, T, N]; a feature token pushes feat_tensor[token] -> [T, N].
Operator tokens pop `arity` operands and push the result. Returns the single remaining
[T, N] tensor, or None on any malformed formula (stack under/overflow, unknown token).
NaN/Inf are scrubbed so a bad sub-expression can't poison the whole run.
"""
from __future__ import annotations

import torch

from .ops import OPS_CONFIG
from .vocab import FORMULA_VOCAB


class StackVM:
    def __init__(self):
        self.feat_offset = FORMULA_VOCAB.operator_offset
        self.op_map = {i + self.feat_offset: cfg[1] for i, cfg in enumerate(OPS_CONFIG)}
        self.arity_map = {i + self.feat_offset: cfg[2] for i, cfg in enumerate(OPS_CONFIG)}

    def execute(self, formula_tokens, feat_tensor):
        stack = []
        try:
            for token in formula_tokens:
                token = int(token)
                if token < self.feat_offset:
                    if token >= feat_tensor.shape[0]:
                        return None
                    stack.append(feat_tensor[token])          # [T, N]
                elif token in self.op_map:
                    arity = self.arity_map[token]
                    if len(stack) < arity:
                        return None
                    args = [stack.pop() for _ in range(arity)]
                    args.reverse()
                    res = self.op_map[token](*args)
                    if torch.isnan(res).any() or torch.isinf(res).any():
                        res = torch.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)
                    stack.append(res)
                else:
                    return None
            return stack[0] if len(stack) == 1 else None
        except Exception:
            return None
