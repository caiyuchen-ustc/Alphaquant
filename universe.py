"""Build the tradeable coin universe from the Binance vision mirror.

There are ~800 USDT perpetual contracts, but most are meme-repriced tickers (1000BONK...),
zombies, or too illiquid to trade. We filter to a clean ~80-150 majors by:
  1. excluding meme-repriced prefixes (1000*, 1000000*, ...),
  2. requiring >= MIN_HISTORY_DAYS of daily bars,
  3. requiring >= MIN_AVG_QUOTE_VOL average daily quote-volume (USD liquidity).

We KEEP contracts that existed historically but may be delisted now (anti-survivorship):
the filter is applied on each contract's own realized history, not on "is it live today".
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from .config import Cfg

_UA = {"User-Agent": "cryptoalpha/universe"}


def list_all_usdt_perps() -> list[str]:
    """Enumerate every *USDT perpetual contract folder on the mirror (paginated S3 XML)."""
    out, token = [], None
    prefix = "data/futures/um/monthly/klines/"
    while True:
        url = f"{Cfg.S3_LIST}?delimiter=/&prefix={prefix}"
        if token:
            url += f"&marker={token}"
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=60) as resp:
            xml = resp.read().decode()
        # crude parse of <Prefix>...</Prefix> entries
        parts = xml.split("<Prefix>")[1:]
        last = None
        for p in parts:
            name = p.split("</Prefix>")[0]
            sym = name[len(prefix):].strip("/")
            last = name
            if sym.endswith("USDT") and "_" not in sym:   # exclude dated delivery (BTCUSDT_250328)
                out.append(sym)
        # pagination
        if "<IsTruncated>true</IsTruncated>" in xml and last:
            token = last
        else:
            break
    return sorted(set(out))


def is_meme(sym: str) -> bool:
    return any(sym.startswith(p) for p in Cfg.EXCLUDE_PREFIXES) or sym in Cfg.EXCLUDE_SYMBOLS


def cache_universe(symbols: list[str]) -> Path:
    Cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    p = Cfg.DATA_DIR / "universe.json"
    p.write_text(json.dumps(symbols, indent=2))
    return p


def load_universe() -> list[str]:
    p = Cfg.DATA_DIR / "universe.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run data_loader with --universe first")
    return json.loads(p.read_text())


if __name__ == "__main__":
    perps = list_all_usdt_perps()
    non_meme = [s for s in perps if not is_meme(s)]
    print(f"total USDT perps: {len(perps)}, non-meme: {len(non_meme)}", flush=True)
    print("sample:", non_meme[:30], flush=True)
