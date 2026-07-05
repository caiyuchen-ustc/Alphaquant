"""Global config for cryptoalpha — Binance USDT-perp cross-sectional symbolic factor mining.

Ported from AlphaGPT (imbue-bit) but with the three fatal bugs fixed:
  bug#1 future-leak  -> cross-sectional standardization (dim=1, coins), see factors.py
  bug#2 dead critic  -> critic value enters the loss as a baseline, see engine.py
  bug#3 no OOS       -> robust reward = min(train,valid) - gap penalty, test touched once

Tensor convention throughout: [T, N]  (T = bars, N = coins).
  - time-series ops (DELAY/DECAY/MAX3) act along dim=0
  - cross-sectional ops (zscore/rank/long-short/jump) act along dim=1
feat_tensor is [N_feat, T, N]; vm indexes feat_tensor[token] -> [T, N].

Paths are relative to cwd=Documents (running `python3 -m cryptoalpha.x` from Documents),
matching the coin/bt/basis_data.py quirk.
"""
from __future__ import annotations

from pathlib import Path


class Cfg:
    # ---- device / paths ----
    DEVICE = "cpu"
    DATA_DIR = Path("data/cryptoalpha")
    REPORTS_DIR = Path("reports/cryptoalpha")

    # ---- data source (Binance vision mirror; unthrottled, no key) ----
    MIRROR = "https://data.binance.vision/data"
    S3_LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
    START_YM = "2020-01"                   # BTCUSDT perp starts 2020-01
    END_YM = "2026-06"
    BENCHMARK = "BTCUSDT"                   # buy-hold comparison

    # ---- universe filtering (dodge meme/zombie contracts) ----
    MIN_HISTORY_DAYS = 365                 # need warmup + train/valid/test
    MIN_AVG_QUOTE_VOL = 10_000_000         # >= $10M avg daily quote volume (liquid)
    EXCLUDE_PREFIXES = ("1000", "1000000", "1MBABYDOGE")   # meme-repriced tickers
    EXCLUDE_SYMBOLS = frozenset({
        # delisted / zero volume
        "BZRXUSDT", "DODOUSDT", "EOSUSDT", "MATICUSDT", "RNDRUSDT",
        # collapsed
        "LUNAUSDT",
        # anomalous volume spike (privacy coin delistings), not real liquidity
        "ZECUSDT",
        # structurally dead (last30 < $5M)
        "AXSUSDT", "DASHUSDT", "ZENUSDT",
    })

    # ---- interval-aware paths and bar metadata ----
    # Each interval gets its own raw CSV cache and panel directory so all
    # frequencies coexist on disk without clobbering each other.
    SUPPORTED_INTERVALS = ("1d", "4h", "1h", "15m")
    BARS_PER_DAY = {"1d": 1, "4h": 6, "1h": 24, "15m": 96}
    BARS_PER_YEAR = {"1d": 365.0, "4h": 365.0 * 6, "1h": 365.0 * 24, "15m": 365.0 * 96}

    # factor warmup in bars (covers the longest rolling window in factors.py)
    WARMUP_BARS = {"1d": 60, "4h": 360, "1h": 1440, "15m": 5760}

    @classmethod
    def raw_dir(cls, interval: str = "1d") -> Path:
        if interval == "1d":
            return cls.DATA_DIR / "raw"           # keep existing 1d layout
        return cls.DATA_DIR / f"raw_{interval}"

    @classmethod
    def panel_dir(cls, interval: str = "1d") -> Path:
        if interval == "1d":
            return cls.DATA_DIR / "panel"         # keep existing 1d layout
        return cls.DATA_DIR / f"panel_{interval}"

    # ---- time splits (by DATE, not random — bug#3). test touched exactly once ----
    TRAIN_RANGE = ("2020-01-01", "2023-06-30")
    VALID_RANGE = ("2023-07-01", "2024-12-31")
    TEST_RANGE = ("2025-01-01", "2026-12-31")

    # ---- portfolio (market-neutral long-short) ----
    TOP_QUANTILE = 0.20                    # long top 20% / short bottom 20%
    COST_PER_SIDE = 0.0005                 # 5 bps one-way perp taker (fee + slippage)

    # ---- factor / warmup (1d defaults; sub-daily uses WARMUP_BARS[interval]) ----
    WARMUP_DAYS = 60                       # kept for 1d backward compat
    MIN_COINS_PER_DAY = 15                 # skip bars with too few valid coins

    # ---- RL (kept small for CPU) ----
    BATCH_SIZE = 256
    TRAIN_STEPS = 300
    MAX_FORMULA_LEN = 12
    D_MODEL = 64
    GAP_PENALTY = 0.5
    VALUE_COEF = 0.5
    ENTROPY_COEF = 0.01
    LR = 1e-3
    MIN_SIGNAL_STD = 1e-4
    SEED = 7

    # ---- annualization (1d default; use BARS_PER_YEAR[interval] for sub-daily) ----
    DAYS_PER_YEAR = 365.0                  # crypto trades 7x24

