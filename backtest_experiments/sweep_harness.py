"""Fast config sweep for backtest_apex: load data + stage-1 signals ONCE,
vary config, re-simulate. Lives OUTSIDE the repo (no repo files changed).
"""
import sys, copy, time
from pathlib import Path
import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import backtest_apex as bt
import config as cfg

SYMS = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "LTC/USD",
        "AVAX/USD", "LINK/USD", "ADA/USD", "BCH/USD", "DOT/USD"]
START = pd.Timestamp("2025-04-01")
END = pd.Timestamp("2026-10-08")
WARM = START - pd.Timedelta(days=3)
CACHE = Path("bt_cache")
OOS = pd.Timestamp("2026-07-19")   # model last-commit cut -> "possibly in-sample" boundary

t0 = time.time()
bars = bt.fetch_alpaca(SYMS, WARM, END, CACHE)
infer, msig = bt.make_infer(cfg.MODEL_PATH)
s1 = {s: bt.stage1_cached(s, d, infer, msig, CACHE, "latest64") for s, d in bars.items()}
MD = bt.MarketData(bars, s1, START, END, "live", list(cfg.DYNAMIC_UNIVERSE_CANDIDATES))
print(f"[harness] data+signals ready in {time.time()-t0:.0f}s", flush=True)

FIELDS = ["MAX_HOLD_HOURS", "PROFIT_TARGET_PCT", "STOP_LOSS_PCT", "SLOW_BLEED_PCT",
          "SLOW_BLEED_MIN_HOURS", "MIN_TRAILING_STOP_PCT", "MAX_TRAILING_STOP_PCT",
          "TRAILING_STOP_ATR_MULTIPLIER", "MIN_HOLD_HOURS_BEFORE_SIGNAL",
          "MAX_OPEN_POSITIONS", "BASE_RISK_PERCENT", "MAX_POSITION_PCT",
          "COOLDOWN_SECONDS_SELL", "MIN_POSITION_USD", "MIN_ORDER_USD"]


def run(**over):
    kill = over.pop("killswitch", "daily")
    buy = over.pop("buy_signal", cfg.BUY_SIGNAL)
    sell = over.pop("sell_signal", cfg.SELL_SIGNAL)
    fee = over.pop("fee_bps", 25.0)
    slip = over.pop("slippage_bps", 5.0)
    saved = {k: getattr(cfg, k) for k in over}
    for k, v in over.items():
        setattr(cfg, k, v)
    try:
        p = bt.Params(equity0=cfg.ACCOUNT_BASE, fee_bps=fee, slippage_bps=slip,
                      fill_model="touch", killswitch=kill, universe_mode="live",
                      buy_signal=buy, sell_signal=sell, start=START, end=END)
        res = bt.Simulator(MD, p).run()
        S = bt.summarize(res)
        S["_res"] = res
        S["_params"] = p
    finally:
        for k, v in saved.items():
            setattr(cfg, k, v)
    return S


def split(S, oos=OOS):
    """(in-sample summary, out-of-sample summary) using the model cutoff."""
    res, p = S["_res"], S["_params"]
    tr = res.trades
    S_is = bt.summarize(bt.SimResult(tr[tr["entry_ts"] < oos] if len(tr) else tr,
                                     res.equity[res.equity.index < oos], res.counters, None, "", p))
    S_oos = bt.summarize(res, t_from=oos)
    return S_is, S_oos


def nullp(S, trials=50, seed=42):
    res, p = S["_res"], S["_params"]
    nd = bt.run_null(MD, s1, p, trials, seed)
    act = res.trades["ret_net"].mean() * 1e4 if len(res.trades) else float("nan")
    nm = nd["mean_net_bps"].mean()
    pval = float((nd["mean_net_bps"] >= act).mean())
    return nm, pval


def line(name, S):
    if not S or not S.get("n_trades"):
        return f"{name:38s} | no trades"
    return (f"{name:38s} | n={S['n_trades']:>6} | win={S['win_rate']*100:5.1f}% | "
            f"ret={S.get('total_return', float('nan'))*100:+7.2f}% | "
            f"net/trade={S['mean_net_bps']:+7.1f}bps | gross={S['mean_gross_bps']:+6.1f} | "
            f"PF={S['profit_factor']:5.2f} | maxDD={S.get('max_dd', float('nan'))*100:+6.2f}%")


if __name__ == "__main__":
    print(line("BASELINE (live killswitch)", run(killswitch="live")))
    print(line("baseline daily-kill", run(killswitch="daily")))
    print(line("baseline kill off", run(killswitch="off")))
