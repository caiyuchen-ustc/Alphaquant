"""Cross-sectional long-short (market-neutral) backtest.

Each day: rank coins by the signal, long the top quantile, short the bottom quantile,
equal-weight within each leg, gross exposure 1.0 per side (weights sum ~0 => neutral).
The signal at day t uses only data up to t; it earns fwd_ret[t] (t->t+1), which is
strictly future relative to t — no look-ahead. Turnover cost is charged on |Δweight|.

Metrics: annualized Sharpe (×√365, crypto trades 7x24), annual return, max drawdown,
plus daily cross-sectional rank-IC (mean and IR) — the cleanest factor-quality signal.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch

from .config import Cfg


@dataclass
class XSResult:
    sharpe: float
    ann_return: float
    ann_vol: float
    max_drawdown: float
    ic_mean: float
    ic_ir: float
    turnover: float
    n_days: int
    exposure: float


def signal_to_weights(signal: torch.Tensor, mask: torch.Tensor, top_q: float) -> torch.Tensor:
    """[T,N] signal -> [T,N] market-neutral weights (long top_q, short bottom_q, equal-weight)."""
    T, N = signal.shape
    w = torch.zeros_like(signal)
    m = mask.bool()
    for t in range(T):
        rm = m[t]
        k = int(rm.sum())
        if k < Cfg.MIN_COINS_PER_DAY:
            continue
        idx = torch.nonzero(rm, as_tuple=True)[0]
        vals = signal[t, idx]
        n_side = max(1, int(k * top_q))
        order = torch.argsort(vals)
        short_idx = idx[order[:n_side]]
        long_idx = idx[order[-n_side:]]
        w[t, long_idx] = 0.5 / n_side       # long leg sums to +0.5
        w[t, short_idx] = -0.5 / n_side     # short leg sums to -0.5  => gross 1.0, net 0
    return w


def _rank_ic(signal: torch.Tensor, fwd: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Daily cross-sectional Spearman-ish rank correlation between signal and fwd return."""
    T = signal.shape[0]
    ics = np.full(T, np.nan)
    m = mask.bool()
    for t in range(T):
        rm = m[t] & torch.isfinite(fwd[t])
        k = int(rm.sum())
        if k < Cfg.MIN_COINS_PER_DAY:
            continue
        s = signal[t, rm]
        f = fwd[t, rm]
        sr = torch.argsort(torch.argsort(s)).float()
        fr = torch.argsort(torch.argsort(f)).float()
        sr = (sr - sr.mean()) / (sr.std() + 1e-9)
        fr = (fr - fr.mean()) / (fr.std() + 1e-9)
        ics[t] = float((sr * fr).mean())
    return ics


def backtest_long_short(signal, fwd_ret, mask, top_q=None, cost=None):
    """Return (XSResult, daily_return_np). fwd_ret must already be look-ahead-safe [T,N]."""
    top_q = top_q or Cfg.TOP_QUANTILE
    cost = Cfg.COST_PER_SIDE if cost is None else cost
    w = signal_to_weights(signal, mask, top_q)
    fwd = torch.nan_to_num(fwd_ret, nan=0.0)

    gross = (w * fwd).sum(dim=1)                          # [T]
    dw = torch.zeros_like(w)
    dw[1:] = torch.abs(w[1:] - w[:-1])
    turnover = dw.sum(dim=1)                              # [T]
    net = gross - turnover * cost
    net = net.cpu().numpy()

    ann = np.sqrt(Cfg.DAYS_PER_YEAR)
    mean, std = net.mean(), net.std(ddof=1)
    sharpe = float(mean / std * ann) if std > 0 else 0.0
    equity = np.cumprod(1.0 + net)
    years = len(net) / Cfg.DAYS_PER_YEAR
    ann_ret = float(equity[-1] ** (1.0 / years) - 1.0) if years > 0 and equity[-1] > 0 else -1.0
    ann_vol = float(std * ann)
    peak = np.maximum.accumulate(equity)
    max_dd = float((equity / peak - 1.0).min())

    ics = _rank_ic(signal, fwd_ret, mask)
    ic_valid = ics[~np.isnan(ics)]
    ic_mean = float(ic_valid.mean()) if len(ic_valid) else 0.0
    ic_ir = float(ic_valid.mean() / (ic_valid.std() + 1e-9) * ann) if len(ic_valid) else 0.0

    res = XSResult(
        sharpe=round(sharpe, 3), ann_return=round(ann_ret, 4), ann_vol=round(ann_vol, 4),
        max_drawdown=round(max_dd, 4), ic_mean=round(ic_mean, 4), ic_ir=round(ic_ir, 3),
        turnover=round(float(turnover.mean()), 4), n_days=len(net),
        exposure=round(float((mask.bool().sum(dim=1) >= Cfg.MIN_COINS_PER_DAY).float().mean()), 3),
    )
    return res, net


def quick_score(signal, fwd_ret, mask, top_q=None, cost=None) -> float:
    """Fast scalar for the RL inner loop: net Sharpe, with degenerate-signal penalties."""
    if signal is None:
        return -5.0
    if float(signal.std()) < Cfg.MIN_SIGNAL_STD:
        return -2.0
    res, _ = backtest_long_short(signal, fwd_ret, mask, top_q, cost)
    return res.sharpe
