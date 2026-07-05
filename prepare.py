"""Shared data preparation: panels -> torch tensors + valid_mask + fwd_ret, sliced by
train/valid/test date ranges. Every module (factors, backtest, engine) consumes this so
masks and splits are identical everywhere.

valid_mask[t,n] is True when coin n on day t is listed AND liquid enough to trade
(rolling quote-volume above threshold) AND past the factor warmup. Cross-sectional stats
and portfolio construction only use valid cells.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .config import Cfg
from .data_loader import load_panel, forward_return
from .factors import compute_feature_tensor


@dataclass
class Split:
    close: torch.Tensor
    high: torch.Tensor
    low: torch.Tensor
    volume: torch.Tensor
    quote_volume: torch.Tensor
    mask: torch.Tensor
    fwd_ret: torch.Tensor
    feat: torch.Tensor           # [N_feat, T, N]
    dates: pd.DatetimeIndex
    symbols: list


def _to_tensor(df: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(df.to_numpy(dtype=np.float64), dtype=torch.float32, device=Cfg.DEVICE)


def build_full():
    """Build the full-history tensors + mask + fwd_ret + feature tensor once."""
    panels = load_panel()
    close = panels["close"]
    dates = close.index
    symbols = list(close.columns)

    fwd = forward_return(close)

    close_t = _to_tensor(close)
    high_t = _to_tensor(panels["high"])
    low_t = _to_tensor(panels["low"])
    vol_t = _to_tensor(panels["volume"])
    qvol_t = _to_tensor(panels["quote_volume"])
    fwd_t = _to_tensor(fwd)

    # valid mask: has price, has next-day return, liquid (trailing 30d quote-vol above thr)
    listed = torch.isfinite(close_t) & (close_t > 0)
    has_fwd = torch.isfinite(fwd_t)
    qv = torch.nan_to_num(qvol_t, nan=0.0)
    # trailing 30d mean quote-volume per coin (dim=0), liquidity gate
    from .factors import _ts_roll_mean
    liq = _ts_roll_mean(qv, 30) >= (Cfg.MIN_AVG_QUOTE_VOL * 0.2)   # softer daily gate
    warm = torch.zeros_like(listed)
    warm[Cfg.WARMUP_DAYS:] = True
    mask = listed & has_fwd & liq & warm

    feat = compute_feature_tensor(close_t, high_t, low_t, vol_t, qvol_t, mask)

    return Split(close_t, high_t, low_t, vol_t, qvol_t, mask, fwd_t, feat, dates, symbols)


def _date_slice(full: Split, start: str, end: str) -> Split:
    d = full.dates
    lo = d.searchsorted(pd.Timestamp(start, tz="UTC"))
    hi = d.searchsorted(pd.Timestamp(end, tz="UTC"), side="right")
    sl = slice(lo, hi)
    return Split(
        close=full.close[sl], high=full.high[sl], low=full.low[sl],
        volume=full.volume[sl], quote_volume=full.quote_volume[sl],
        mask=full.mask[sl], fwd_ret=full.fwd_ret[sl], feat=full.feat[:, sl, :],
        dates=d[sl], symbols=full.symbols,
    )


def splits():
    """Return (full, train, valid, test) Split objects using Cfg date ranges."""
    full = build_full()
    train = _date_slice(full, *Cfg.TRAIN_RANGE)
    valid = _date_slice(full, *Cfg.VALID_RANGE)
    test = _date_slice(full, *Cfg.TEST_RANGE)
    return full, train, valid, test
