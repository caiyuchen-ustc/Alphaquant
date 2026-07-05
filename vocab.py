"""Token vocabulary: features (indices into feat_tensor) + operators.

A formula is a postfix (RPN) token sequence. Tokens < feature_count are features; tokens
>= feature_count are operators. FEATURE_NAMES order MUST match factors.compute_feature_tensor.
"""
from __future__ import annotations

from dataclasses import dataclass

from .ops import OPS_CONFIG

FEATURE_NAMES = (
    "MOM_30", "MOM_7", "REV_1", "REV_3", "VOL_30", "QVOL_TREND",
    "ILLIQ", "MAGAP_10", "MAGAP_30", "HLRANGE", "RSI_14", "VOL_TREND",
)


@dataclass(frozen=True)
class FormulaVocab:
    feature_names: tuple
    operator_names: tuple

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def operator_offset(self) -> int:
        return self.feature_count

    @property
    def token_names(self) -> tuple:
        return self.feature_names + self.operator_names

    @property
    def size(self) -> int:
        return len(self.token_names)


FORMULA_VOCAB = FormulaVocab(
    feature_names=FEATURE_NAMES,
    operator_names=tuple(cfg[0] for cfg in OPS_CONFIG),
)
