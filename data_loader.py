"""Download Binance USDT-perp klines from data.binance.vision and align into [T, N] panels.

Supports multiple intervals (1d, 4h, 1h) — each gets its own raw CSV cache and panel dir
so all frequencies coexist on disk without clobbering each other.

Monthly kline CSVs have 12 columns (header row present on futures archives):
  open_time, open, high, low, close, volume, close_time, quote_volume, count,
  taker_buy_volume, taker_buy_quote_volume, ignore
open_time is ms or us (older/newer archives) -> normalized via _norm_ms.

Usage:
  python3 -m cryptoalpha.data_loader --build --interval 4h
  python3 -m cryptoalpha.data_loader --build --interval 1h
  python3 -m cryptoalpha.data_loader --build            # defaults to 1d
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import Cfg
from .universe import list_all_usdt_perps, is_meme, cache_universe, load_universe

FIELDS = ("open", "high", "low", "close", "volume", "quote_volume")


def _norm_ms(v: int) -> int:
    if v > 1_000_000_000_000_000:   # microseconds
        return v // 1000
    return v


def _months(start_ym: str, end_ym: str) -> list[str]:
    sy, sm = map(int, start_ym.split("-"))
    ey, em = map(int, end_ym.split("-"))
    out, y, m = [], sy, sm
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def _download_month(sym: str, ym: str, interval: str = "1d", timeout: int = 120) -> Optional[Path]:
    raw_dir = Cfg.raw_dir(interval)
    csv_path = raw_dir / sym / f"{sym}-{interval}-{ym}.csv"
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{Cfg.MIRROR}/futures/um/monthly/klines/{sym}/{interval}/{sym}-{interval}-{ym}.zip"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "cryptoalpha"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            blob = resp.read()
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            name = [n for n in z.namelist() if n.endswith(".csv")][0]
            csv_path.write_bytes(z.read(name))
        return csv_path
    except Exception:
        return None   # 404 for months before listing — expected, skip silently


def download_symbol(sym: str, interval: str = "1d") -> Optional[pd.DataFrame]:
    """Download+concat all months for one symbol; cache a per-symbol parquet."""
    raw_dir = Cfg.raw_dir(interval)
    cache = raw_dir / f"{sym}.parquet"
    if cache.exists() and cache.stat().st_size > 0:
        return pd.read_parquet(cache)
    frames = []
    for ym in _months(Cfg.START_YM, Cfg.END_YM):
        p = _download_month(sym, ym, interval)
        if p is None:
            continue
        first = p.open().readline()
        header = 0 if "open_time" in first.lower() else None
        df = pd.read_csv(p, header=header, usecols=range(9),
                         names=["open_time", "open", "high", "low", "close",
                                "volume", "close_time", "quote_volume", "count"])
        frames.append(df)
    if not frames:
        return None
    raw = pd.concat(frames, ignore_index=True)
    raw["open_time"] = raw["open_time"].astype("int64").map(_norm_ms)
    raw = raw.drop_duplicates("open_time").sort_values("open_time")
    raw["ts"] = pd.to_datetime(raw["open_time"], unit="ms", utc=True)
    if interval == "1d":
        raw["ts"] = raw["ts"].dt.normalize()   # floor to midnight for daily
    for c in FIELDS:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    out = raw.set_index("ts")[list(FIELDS)].dropna(subset=["close"])
    if len(out):
        out.to_parquet(cache)
    return out


def build_universe(min_history=None, min_vol=None, workers=16) -> list[str]:
    """Download all non-meme perps (1d, concurrent), filter by history + liquidity.

    Universe discovery always uses 1d data (faster, parquets already cached).
    Filter criteria:
    - >= MIN_HISTORY_DAYS of daily bars
    - trailing 365d avg quote-volume >= MIN_AVG_QUOTE_VOL
    - trailing 30d avg quote-volume > 0 (not delisted)
    """
    min_history = min_history or Cfg.MIN_HISTORY_DAYS
    min_vol = min_vol or Cfg.MIN_AVG_QUOTE_VOL

    perps = [s for s in list_all_usdt_perps() if not is_meme(s)]
    print(f"screening {len(perps)} non-meme perps with {workers} workers...", flush=True)
    kept, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(download_symbol, sym, "1d"): sym for sym in perps}
        for fut in as_completed(futs):
            sym = futs[fut]
            done += 1
            try:
                df = fut.result()
            except Exception:
                df = None
            if df is not None and len(df) >= min_history:
                if df["quote_volume"].tail(30).mean() <= 0:
                    pass
                elif df["quote_volume"].tail(365).mean() >= min_vol:
                    kept.append(sym)
            if done % 50 == 0:
                print(f"[{done}/{len(perps)}] scanned, {len(kept)} kept", flush=True)
    kept.sort()
    cache_universe(kept)
    print(f"universe: {len(kept)} liquid coins with >= {min_history}d history", flush=True)
    return kept


def build_panel(symbols=None, interval: str = "1d", workers: int = 16) -> dict[str, pd.DataFrame]:
    """Download sub-daily klines for all universe symbols and align into [T, N] panels."""
    symbols = symbols or load_universe()
    warmup = Cfg.WARMUP_BARS.get(interval, Cfg.WARMUP_DAYS)
    panel_dir = Cfg.panel_dir(interval)

    print(f"building {interval} panel for {len(symbols)} coins...", flush=True)

    per_field: dict[str, dict] = {f: {} for f in FIELDS}
    kept = []

    def _load(sym):
        return sym, download_symbol(sym, interval)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_load, sym): sym for sym in symbols}
        done = 0
        for fut in as_completed(futs):
            sym = futs[fut]
            done += 1
            try:
                _, df = fut.result()
            except Exception:
                df = None
            if df is not None and len(df) >= warmup:
                kept.append(sym)
                for f in FIELDS:
                    per_field[f][sym] = df[f]
            if done % 20 == 0:
                print(f"  [{done}/{len(symbols)}] loaded", flush=True)

    if not kept:
        raise RuntimeError("no symbols with data")

    kept.sort()
    panels = {}
    panel_dir.mkdir(parents=True, exist_ok=True)
    # align on common timestamp index
    idx = pd.DatetimeIndex(
        sorted(set().union(*[set(per_field["close"][s].index) for s in kept]))
    )
    for f in FIELDS:
        wide = pd.DataFrame({s: per_field[f][s] for s in kept}, index=idx)
        wide = wide.sort_index()
        panels[f] = wide
        wide.to_parquet(panel_dir / f"{f}.parquet")

    n_bars = panels["close"].shape[0]
    print(f"panel [{interval}]: {n_bars} bars x {len(kept)} coins -> {panel_dir}", flush=True)
    return panels


def load_panel(interval: str = "1d") -> dict[str, pd.DataFrame]:
    out = {}
    panel_dir = Cfg.panel_dir(interval)
    for f in FIELDS:
        p = panel_dir / f"{f}.parquet"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing — run `python3 -m cryptoalpha.data_loader --build --interval {interval}`"
            )
        out[f] = pd.read_parquet(p)
    return out


def forward_return(close: pd.DataFrame) -> pd.DataFrame:
    """Look-ahead-safe next-bar return: fwd_ret[t] = close[t+1]/close[t]-1."""
    return close.shift(-1) / close - 1.0


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--universe", action="store_true", help="download+filter 1d universe")
    p.add_argument("--build", action="store_true", help="build aligned [T,N] panels")
    p.add_argument("--interval", default="1d", choices=Cfg.SUPPORTED_INTERVALS,
                   help="kline interval (default: 1d)")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    if args.universe:
        build_universe()
    if args.build:
        panels = build_panel(interval=args.interval, workers=args.workers)
        fwd = forward_return(panels["close"])
        cov = panels["close"].notna().mean(axis=1)
        print(f"fwd_ret {fwd.shape}, mean bar-coverage {cov.mean():.1%}, "
              f"date range {panels['close'].index[0]}..{panels['close'].index[-1]}")
