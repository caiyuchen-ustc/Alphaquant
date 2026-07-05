"""Download Binance USDT-perp daily klines from data.binance.vision and align them into
[T, N] panels (T = UTC days, N = coins).

Monthly kline CSVs have 12 columns (header row present on futures archives):
  open_time, open, high, low, close, volume, close_time, quote_volume, count,
  taker_buy_volume, taker_buy_quote_volume, ignore
open_time is ms or us (older/newer archives) -> normalized via _norm_ms (reused convention
from coin/bt/basis_data.py). We keep close (returns) + quote_volume (liquidity filter).

Downloads land under Documents/data/cryptoalpha/ (cwd=Documents when run as a module).
"""
from __future__ import annotations

import datetime as dt
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


def _download_month(sym: str, ym: str, timeout: int = 120) -> Optional[Path]:
    csv_path = Cfg.RAW_DIR / sym / f"{sym}-1d-{ym}.csv"
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{Cfg.MIRROR}/futures/um/monthly/klines/{sym}/1d/{sym}-1d-{ym}.zip"
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


def download_symbol(sym: str) -> Optional[pd.DataFrame]:
    """Download+concat all months for one symbol; cache a per-symbol parquet. Returns daily df."""
    cache = Cfg.RAW_DIR / f"{sym}.parquet"
    if cache.exists() and cache.stat().st_size > 0:
        return pd.read_parquet(cache)
    frames = []
    for ym in _months(Cfg.START_YM, Cfg.END_YM):
        p = _download_month(sym, ym)
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
    raw["ts"] = pd.to_datetime(raw["open_time"], unit="ms", utc=True).dt.normalize()
    for c in FIELDS:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    out = raw.set_index("ts")[list(FIELDS)].dropna(subset=["close"])
    if len(out):
        out.to_parquet(cache)
    return out


def build_universe(min_history=None, min_vol=None, workers=16) -> list[str]:
    """Download all non-meme perps (concurrently), then filter by realized history + liquidity.

    Filter criteria:
    - >= MIN_HISTORY_DAYS of daily bars (ensures enough warmup + backtest coverage)
    - trailing 365d average quote-volume >= MIN_AVG_QUOTE_VOL ($50M): liquid enough to trade
      *right now* (or at delisting time). Using recent avg rather than full-history median
      keeps coins that grew into liquidity (2024-25 listings) while still catching zombies
      whose volume dried up.
    """
    min_history = min_history or Cfg.MIN_HISTORY_DAYS
    min_vol = min_vol or Cfg.MIN_AVG_QUOTE_VOL

    perps = [s for s in list_all_usdt_perps() if not is_meme(s)]
    print(f"screening {len(perps)} non-meme perps with {workers} workers...", flush=True)
    kept = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(download_symbol, sym): sym for sym in perps}
        for fut in as_completed(futs):
            sym = futs[fut]
            done += 1
            try:
                df = fut.result()
            except Exception:
                df = None
            if df is not None and len(df) >= min_history:
                # must still be actively trading (not delisted / zero-volume zombie)
                if df["quote_volume"].tail(30).mean() <= 0:
                    pass
                # trailing 365d avg — coins with <365d use all available days
                elif df["quote_volume"].tail(365).mean() >= min_vol:
                    kept.append(sym)
            if done % 50 == 0:
                print(f"[{done}/{len(perps)}] scanned, {len(kept)} kept", flush=True)
    kept.sort()
    cache_universe(kept)
    print(f"universe: {len(kept)} liquid coins with >= {min_history}d history", flush=True)
    return kept


def build_panel(symbols=None) -> dict[str, pd.DataFrame]:
    """Assemble per-field [T,N] panels aligned on the UTC daily calendar."""
    symbols = symbols or load_universe()
    per_field = {f: {} for f in FIELDS}
    kept = []
    for sym in symbols:
        df = download_symbol(sym)
        if df is None or len(df) < Cfg.WARMUP_DAYS:
            continue
        kept.append(sym)
        for f in FIELDS:
            per_field[f][sym] = df[f]
    if not kept:
        raise RuntimeError("no symbols with data")
    panels = {}
    Cfg.PANEL_DIR.mkdir(parents=True, exist_ok=True)
    for f in FIELDS:
        wide = pd.DataFrame(per_field[f]).sort_index().reindex(columns=kept)
        panels[f] = wide
        wide.to_parquet(Cfg.PANEL_DIR / f"{f}.parquet")
    print(f"panel: {panels['close'].shape[0]} days x {len(kept)} coins -> {Cfg.PANEL_DIR}",
          flush=True)
    return panels


def load_panel() -> dict[str, pd.DataFrame]:
    out = {}
    for f in FIELDS:
        p = Cfg.PANEL_DIR / f"{f}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing — run `python3 -m cryptoalpha.data_loader --build`")
        out[f] = pd.read_parquet(p)
    return out


def forward_return(close: pd.DataFrame) -> pd.DataFrame:
    """Look-ahead-safe next-day return: fwd_ret[t] = close[t+1]/close[t]-1 (enter t, exit t+1)."""
    return close.shift(-1) / close - 1.0


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--universe", action="store_true", help="download+filter universe")
    p.add_argument("--build", action="store_true", help="build aligned [T,N] panels")
    args = p.parse_args()
    if args.universe:
        build_universe()
    if args.build:
        panels = build_panel()
        fwd = forward_return(panels["close"])
        cov = panels["close"].notna().mean(axis=1)
        print(f"fwd_ret {fwd.shape}, mean daily coin-coverage {cov.mean():.1%}, "
              f"date range {panels['close'].index[0].date()}..{panels['close'].index[-1].date()}")
