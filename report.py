"""Report + four honesty checks for the mined formula.

Any check failing = still p-hacking:
  1. train/valid/test Sharpe & IC same order of magnitude, same sign.
  2. vs buy-hold BTC: market-neutral vol far below BTC; test Sharpe>0 and low BTC correlation.
  3. random-formula baseline: learned formula must sit in the right tail (>95th pct).
  4. per-year honesty + cost sensitivity.
Also declares residual survivorship bias.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from .backtest import backtest_long_short
from .config import Cfg
from .engine import readable
from .prepare import splits
from .vm import StackVM


def _btc_buyhold(split):
    """Buy-hold BTC daily returns over the split window (benchmark)."""
    if Cfg.BENCHMARK not in split.symbols:
        return None
    j = split.symbols.index(Cfg.BENCHMARK)
    c = split.close[:, j].cpu().numpy()
    r = np.zeros(len(c)); r[1:] = c[1:] / c[:-1] - 1.0
    r = np.nan_to_num(r)
    ann = np.sqrt(Cfg.DAYS_PER_YEAR)
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    return {
        "ann_return": float(eq[-1] ** (Cfg.DAYS_PER_YEAR / max(len(r), 1)) - 1) if eq[-1] > 0 else -1,
        "ann_vol": float(r.std(ddof=1) * ann),
        "sharpe": float(r.mean() / (r.std(ddof=1) + 1e-9) * ann),
        "max_dd": float((eq / peak - 1).min()),
        "daily": r,
    }


def _random_baseline(split, vm, n=100, seed=0):
    """Distribution of test Sharpe for random formulas — the 'is it luck' control."""
    from .vocab import FORMULA_VOCAB
    g = torch.Generator().manual_seed(seed)
    V = FORMULA_VOCAB.size
    sharpes = []
    for _ in range(n):
        L = int(torch.randint(2, Cfg.MAX_FORMULA_LEN + 1, (1,), generator=g))
        f = torch.randint(0, V, (L,), generator=g).tolist()
        sig = vm.execute(f, split.feat)
        if sig is None:
            continue
        res, _ = backtest_long_short(sig, split.fwd_ret, split.mask)
        sharpes.append(res.sharpe)
    return np.array(sharpes)


def _per_year(split, daily):
    import pandas as pd
    s = pd.Series(daily, index=split.dates)
    rows = {}
    ann = np.sqrt(Cfg.DAYS_PER_YEAR)
    for yr, g in s.groupby(s.index.year):
        v = g.to_numpy()
        rows[int(yr)] = {"days": len(v), "ann_ret": round(float((1+v).prod()**(Cfg.DAYS_PER_YEAR/len(v))-1), 4),
                         "sharpe": round(float(v.mean()/(v.std(ddof=1)+1e-9)*ann), 2)}
    return rows


def build_report():
    with open(Cfg.REPORTS_DIR / "search_result.json") as f:
        sr = json.load(f)
    winner = sr["best_formula"]
    full, train, valid, test = splits()
    vm = StackVM()

    sig_test = vm.execute(winner, test.feat)
    res_test, daily_test = backtest_long_short(sig_test, test.fwd_ret, test.mask)
    btc = _btc_buyhold(test)
    rand = _random_baseline(test, vm, n=100)
    pct = float((rand < res_test.sharpe).mean() * 100) if len(rand) else float("nan")
    per_year = _per_year(test, daily_test)

    # cost sensitivity on test
    cost_rows = {}
    for bps in (5, 10, 20, 30):
        r, _ = backtest_long_short(sig_test, test.fwd_ret, test.mask, cost=bps / 10000.0)
        cost_rows[f"{bps}bps"] = {"sharpe": r.sharpe, "ann_ret": r.ann_return}

    corr = float(np.corrcoef(daily_test, btc["daily"])[0, 1]) if btc else float("nan")

    L = []
    L.append("# cryptoalpha:Binance 永续横截面因子挖掘 — 回测报告\n")
    L.append(f"数据: {len(full.symbols)} 个流动 USDT 永续, {str(full.dates[0].date())}..{str(full.dates[-1].date())}。")
    L.append(f"多空对冲(market-neutral),top/bottom {Cfg.TOP_QUANTILE:.0%},单边成本 {Cfg.COST_PER_SIDE*1e4:.0f}bps。")
    L.append(f"训练/验证/测试按日期切分,test 仅在最后评估一次。\n")
    L.append(f"**挖出的最优公式**: `{sr['best_formula_readable']}`\n")

    L.append("## 1. train/valid/test 三段对比(过拟合检验)\n```")
    L.append(f"{'split':>6} {'sharpe':>8} {'ic_mean':>9} {'ann_ret':>9} {'max_dd':>8}")
    for s in ("train", "valid", "test"):
        r = sr[s]
        L.append(f"{s:>6} {r['sharpe']:>+8.2f} {r['ic_mean']:>+9.4f} {r['ann_return']:>+9.1%} {r['max_drawdown']:>8.1%}")
    L.append("```")
    consistent = (np.sign(sr['train']['ic_mean']) == np.sign(sr['test']['ic_mean'])) and sr['test']['sharpe'] > 0
    L.append(f"→ {'✅ test 与 train 同号且为正,未见明显过拟合' if consistent else '⚠️ test 与训练不一致,仍疑过拟合'}\n")

    L.append("## 2. 对照买入持有 BTC(alpha vs beta)\n```")
    if btc:
        L.append(f"{'':>12}{'carry(LS)':>12}{'BTC hold':>12}")
        L.append(f"{'ann_return':>12}{res_test.ann_return:>+12.1%}{btc['ann_return']:>+12.1%}")
        L.append(f"{'ann_vol':>12}{res_test.ann_vol:>12.1%}{btc['ann_vol']:>12.1%}")
        L.append(f"{'sharpe':>12}{res_test.sharpe:>+12.2f}{btc['sharpe']:>+12.2f}")
        L.append(f"{'max_dd':>12}{res_test.max_drawdown:>12.1%}{btc['max_dd']:>12.1%}")
        L.append(f"corr(LS, BTC) = {corr:+.3f}")
    L.append("```\n")

    L.append("## 3. 随机公式对照基线(是不是运气)\n```")
    if len(rand):
        L.append(f"随机公式 test Sharpe: mean={rand.mean():+.2f} std={rand.std():.2f} "
                 f"p50={np.median(rand):+.2f} p95={np.percentile(rand,95):+.2f}")
        L.append(f"学到的公式 Sharpe={res_test.sharpe:+.2f} → 位于随机分布 {pct:.0f} 百分位")
        L.append("✅ 显著优于随机" if pct >= 95 else "⚠️ 未显著超过随机,edge 可能是运气")
    L.append("```\n")

    L.append("## 4. 分年份 + 成本敏感性(test)\n```")
    L.append("per-year:")
    for yr, d in per_year.items():
        L.append(f"  {yr}: Sharpe {d['sharpe']:+.2f}  annRet {d['ann_ret']:+.1%}  ({d['days']}d)")
    L.append("cost sensitivity:")
    for k, v in cost_rows.items():
        L.append(f"  {k}: Sharpe {v['sharpe']:+.2f}  annRet {v['ann_ret']:+.1%}")
    L.append("```\n")

    L.append("## 5. 已知偏差与结论\n")
    L.append("- **幸存者偏差**:已纳入历史退市合约缓解,但 universe 仍偏向存活至今的币,残留正偏。")
    L.append("- **换手成本**:多空每日再平衡换手高,对成本敏感(见第4节)——实盘须用 maker 降本。")
    L.append(f"- 与原 AlphaGPT 不同:因子横截面标准化(无未来函数)、critic 真正进 loss、reward 用 min(train,valid),test 只评一次。\n")

    report = "\n".join(L)
    (Cfg.REPORTS_DIR / "final_report.md").write_text(report)
    print(report)
    return report


if __name__ == "__main__":
    build_report()
