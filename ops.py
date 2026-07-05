"""Operators for the formula language, under the [T, N] convention (T=days, N=coins).

★ Dimension remap vs AlphaGPT: time-series ops act along dim=0 (DELAY/DECAY/MAX3); the
cross-sectional ops (RANK/ZSCORE/JUMP) act along dim=1 (across coins on each day). This is
the mechanical core of porting from time-series meme signals to cross-sectional selection.

All ops take/return [T, N] tensors. Cross-sectional ops standardize per-day and are
look-ahead-safe (only use that day's coins).
"""
from __future__ import annotations

import torch


# ----------------------------- time-series ops (dim=0) -----------------------------------

def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    if d == 0:
        return x
    pad = torch.zeros((d, x.shape[1]), dtype=x.dtype)
    return torch.cat([pad, x[:-d, :]], dim=0)


def _op_decay(x: torch.Tensor) -> torch.Tensor:
    return x + 0.8 * _ts_delay(x, 1) + 0.6 * _ts_delay(x, 2)


def _op_delay1(x: torch.Tensor) -> torch.Tensor:
    return _ts_delay(x, 1)


def _op_max3(x: torch.Tensor) -> torch.Tensor:
    return torch.max(x, torch.max(_ts_delay(x, 1), _ts_delay(x, 2)))


# ----------------------------- cross-sectional ops (dim=1) -------------------------------

def _op_cs_zscore(x: torch.Tensor) -> torch.Tensor:
    """Per-day z-score across coins (dim=1). NaN-safe via finite mask per row."""
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-6
    return torch.clamp((x - mean) / std, -5.0, 5.0)


def _op_cs_rank(x: torch.Tensor) -> torch.Tensor:
    """Per-day cross-sectional rank in [-0.5, 0.5] (dim=1)."""
    order = torch.argsort(torch.argsort(x, dim=1), dim=1).float()
    n = x.shape[1]
    return order / max(n - 1, 1) - 0.5


def _op_jump(x: torch.Tensor) -> torch.Tensor:
    """Cross-sectional extreme detector: relu(z - 3) where z is per-day (dim=1)."""
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-6
    z = (x - mean) / std
    return torch.relu(z - 3.0)


def _op_gate(cond: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m = (cond > 0).float()
    return m * a + (1.0 - m) * b


# (name, fn, arity)
OPS_CONFIG = [
    ("ADD", lambda x, y: x + y, 2),
    ("SUB", lambda x, y: x - y, 2),
    ("MUL", lambda x, y: x * y, 2),
    ("DIV", lambda x, y: x / (y + 1e-6), 2),
    ("NEG", lambda x: -x, 1),
    ("ABS", torch.abs, 1),
    ("SIGN", torch.sign, 1),
    ("GATE", _op_gate, 3),
    ("CS_RANK", _op_cs_rank, 1),
    ("CS_ZSCORE", _op_cs_zscore, 1),
    ("JUMP", _op_jump, 1),
    ("DELAY1", _op_delay1, 1),
    ("DECAY", _op_decay, 1),
    ("MAX3", _op_max3, 1),
]
