"""Cross-sectional crypto factors + cross-sectional standardization.

★ This is where bug#1 (future-leak) is fixed. AlphaGPT's robust_norm standardized along
the TIME axis using the whole history's median/MAD — so factor value at day t used data
from days after t. Here every factor is standardized CROSS-SECTIONALLY: on each day t we
z-score / rank across the coins alive that day (dim=1). That uses only day-t information,
so it is look-ahead-safe by construction.

Tensor convention: everything is [T, N] (T=days, N=coins). Time-series rolling ops act
along dim=0; cross-sectional standardization acts along dim=1. A valid_mask [T,N] marks
which (day, coin) cells are tradeable (listed + liquid); invalid cells are excluded from
cross-sectional statistics and set to 0 in the output.
"""
from __future__ import annotations

import torch

from .vocab import FEATURE_NAMES


# ----------------------------- cross-sectional standardizers (bug#1 fix) -----------------

def cross_sectional_zscore(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-day z-score across coins (dim=1). Only valid coins contribute to mean/std."""
    m = mask.bool()
    xm = torch.where(m, x, torch.nan)
    mean = torch.nanmean(xm, dim=1, keepdim=True)
    var = torch.nanmean((xm - mean) ** 2, dim=1, keepdim=True)
    std = torch.sqrt(var + 1e-12)
    z = (x - mean) / std
    z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    z = torch.clamp(z, -5.0, 5.0)
    return torch.where(m, z, torch.zeros_like(z))


def cross_sectional_rank(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-day rank across coins mapped to [-0.5, 0.5]. Robust to outliers."""
    m = mask.bool()
    T, N = x.shape
    out = torch.zeros_like(x)
    for t in range(T):
        row_mask = m[t]
        k = int(row_mask.sum())
        if k < 2:
            continue
        vals = x[t, row_mask]
        order = torch.argsort(torch.argsort(vals).float())
        ranks = order / (k - 1) - 0.5           # [-0.5, 0.5]
        out[t, row_mask] = ranks
    return out


# ----------------------------- time-series helpers (dim=0) -------------------------------

def _ts_roll_sum(x: torch.Tensor, w: int) -> torch.Tensor:
    """Trailing rolling sum over w days along dim=0 (past-only, left-padded)."""
    T, N = x.shape
    xf = torch.nan_to_num(x, nan=0.0)
    pad = torch.zeros((w - 1, N), dtype=x.dtype)
    xp = torch.cat([pad, xf], dim=0)
    return xp.unfold(0, w, 1).sum(dim=-1)


def _ts_roll_mean(x: torch.Tensor, w: int) -> torch.Tensor:
    return _ts_roll_sum(x, w) / w


def _ts_roll_std(x: torch.Tensor, w: int) -> torch.Tensor:
    m = _ts_roll_mean(x, w)
    m2 = _ts_roll_mean(x * x, w)
    return torch.sqrt(torch.clamp(m2 - m * m, min=1e-12))


def _log_ret(close: torch.Tensor) -> torch.Tensor:
    prev = torch.roll(close, 1, dims=0)
    prev[0] = close[0]
    return torch.log(close / (prev + 1e-12) + 1e-12)


# ----------------------------- raw crypto factors (before standardization) ----------------

def _raw_factors(close, high, low, volume, quote_volume) -> dict[str, torch.Tensor]:
    ret = _log_ret(close)
    out = {}
    # momentum: cumulative log-return over trailing window (crypto uses shorter than equities)
    out["MOM_30"] = _ts_roll_sum(ret, 30)
    out["MOM_7"] = _ts_roll_sum(ret, 7)
    # reversal: negative recent return
    out["REV_1"] = -ret
    out["REV_3"] = -_ts_roll_sum(ret, 3)
    # realized volatility (low-vol vs high-vol cross-sectional spread)
    out["VOL_30"] = _ts_roll_std(ret, 30)
    # quote-volume trend: recent vs baseline liquidity (activity surge)
    qv = torch.log1p(quote_volume)
    out["QVOL_TREND"] = _ts_roll_mean(qv, 7) - _ts_roll_mean(qv, 30)
    # Amihud illiquidity: |ret| / quote_volume, higher = more illiquid
    out["ILLIQ"] = _ts_roll_mean(torch.abs(ret) / (quote_volume + 1.0), 14)
    # moving-average gap: price deviation from its own MA (trend/pump)
    ma10 = _ts_roll_mean(close, 10)
    ma30 = _ts_roll_mean(close, 30)
    out["MAGAP_10"] = (close - ma10) / (ma10 + 1e-12)
    out["MAGAP_30"] = (close - ma30) / (ma30 + 1e-12)
    # high-low range (intraday volatility proxy)
    out["HLRANGE"] = _ts_roll_mean((high - low) / (close + 1e-12), 14)
    # RSI-14 (de-centered)
    up = torch.relu(close - torch.roll(close, 1, dims=0))
    dn = torch.relu(torch.roll(close, 1, dims=0) - close)
    up[0] = 0; dn[0] = 0
    avg_up = _ts_roll_mean(up, 14)
    avg_dn = _ts_roll_mean(dn, 14)
    rs = (avg_up + 1e-12) / (avg_dn + 1e-12)
    out["RSI_14"] = (100 - 100 / (1 + rs)) / 50 - 1.0
    # volume trend (raw volume acceleration)
    lv = torch.log1p(volume)
    out["VOL_TREND"] = _ts_roll_mean(lv, 7) - _ts_roll_mean(lv, 30)
    return out


def compute_feature_tensor(close, high, low, volume, quote_volume, mask) -> torch.Tensor:
    """Return [N_feat, T, N] — each factor time-series-computed then cross-sectionally z-scored."""
    raw = _raw_factors(close, high, low, volume, quote_volume)
    feats = []
    for name in FEATURE_NAMES:
        z = cross_sectional_zscore(raw[name], mask)
        feats.append(z)
    return torch.stack(feats, dim=0)


def single_factor(name: str, close, high, low, volume, quote_volume, mask) -> torch.Tensor:
    """One standardized factor [T,N] by name (for the single-factor baseline test)."""
    raw = _raw_factors(close, high, low, volume, quote_volume)
    return cross_sectional_zscore(raw[name], mask)
