"""Funding-carry strategy: a market-neutral, multi-day book on perp funding.

This is a deliberately DIFFERENT shape from the repo's incumbent bot
(2-hour holds, long-only, single-symbol, OHLCV/ML). It is the one candidate that
the research (`research/market_neutral_funding.py`, `research/RESULTS.md`) found
to be net-positive at realistic costs, with a timing-alpha null p-value of 0.00:

  * **Signal**: rank the universe by trailing 7-day cumulative funding
    (``lookback`` hours, summed from the 8h settlements). The book goes **long
    the lowest-funding names and short the highest-funding names** -- i.e. it
    collects the funding cashflow (short legs earn funding when funding > 0)
    and, empirically, also earns a robust short-horizon price reversal as
    crowded-funding names underperform.
  * **Book**: equal-weight, market-neutral long-top-K / short-bottom-K, so
    crypto beta cancels and the P&L is a genuine timing/carry edge.
  * **Cadence**: rebalance every ``hold`` hours (default 24h), NOT every bar.
  * **Costs**: modelled per fill at Binance USDⓈ-M fees (taker 5 bps / maker
    2 bps). At the incumbent venue's Alpaca tier-1 taker (25 bps) the edge does
    not survive -- this strategy belongs on a perp venue.

The functions here are pure (no network, no venue SDK) so they can be unit
tested and reasoned about; data loading is injected. See ``research/RESULTS.md``
for the walk-forward evidence and the honest caveats (regime concentration).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ── Defaults (chosen by out-of-sample walk-forward, not in-sample tuning) ──────
# FROZEN RULE (2026-10-09): K=3, 7-day funding lookback, 24h rebalance. Do NOT
# tune these further -- they are fixed for forward/backward validation. Changing
# any of them invalidates the out-of-sample tests (see research/RESULTS.md).
LOOKBACK_HOURS = 168     # 7 days of funding history  [FROZEN]
HOLD_HOURS = 24          # rebalance cadence           [FROZEN]
TOP_K = 3                # names per side              [FROZEN]
FUNDING_INTERVAL_HOURS = 8

# Frozen universe: liquid Binance USDT-M perps with long funding history.
# Do NOT add/remove names -- a changing universe is another form of tuning.
DEFAULT_UNIVERSE = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX",
    "LINK", "DOT", "LTC", "BCH", "TRX", "ATOM", "UNI",
]

# Binance USDⓈ-M fee schedule (VIP0), bps per fill.
TAKER_BPS = 5.0
MAKER_BPS = 2.0
# Incumbent venue (Alpaca tier-1 crypto taker) -- kept to show the venue matters.
ALPACA_TAKER_BPS = 25.0


@dataclass
class RebalanceOrder:
    symbol: str
    side: str          # "buy" | "sell"
    target_weight: float
    delta_weight: float


def cross_sectional_z(W: pd.DataFrame) -> pd.DataFrame:
    """z-score each row (timestamp) across symbols; NaN-safe."""
    mu = W.mean(axis=1)
    sd = W.std(axis=1).replace(0.0, np.nan)
    return W.sub(mu, axis=0).div(sd, axis=0)


def funding_carry_score(funding_hourly: pd.DataFrame,
                        lookback: int = LOOKBACK_HOURS) -> pd.DataFrame:
    """Score panel (higher = better long).

    ``funding_hourly`` is an hourly panel of the 8h funding rate (the live feed
    repeats the last settlement until the next one). Cumulative funding over
    ``lookback`` hours is the crowding/positioning cost of holding each name;
    we short the crowded (high) names and long the cheap (low) names, then
    cross-sectionally z-score so symbols are comparable bar-to-bar.
    """
    cum = funding_hourly.rolling(lookback, min_periods=lookback // 2).sum()
    return -cross_sectional_z(cum)


def select_book(score_row: pd.Series, K: int = TOP_K,
                long_only: bool = False) -> dict[str, float]:
    """Turn one timestamp's scores into equal-weight, dollar-neutral targets.

    Long the top-K scores, short the bottom-K. Each leg carries ``1/K``, so the
    long side sums to +1 and the short side to -1 (gross exposure 2, net 0) --
    this matches the research harness, which averages each side and differences
    them. A long-only book has a single +1 side.
    """
    s = score_row.dropna()
    if len(s) < (K if long_only else 2 * K):
        return {}
    ordered = s.sort_values()
    longs = list(ordered.index[-K:])
    shorts = [] if long_only else list(ordered.index[:K])
    leg = 1.0 / K
    w = {sym: leg for sym in longs}
    for sym in shorts:
        w[sym] = -leg
    return w


def target_weights(score: pd.DataFrame, K: int = TOP_K,
                   long_only: bool = False) -> pd.DataFrame:
    """Full target-weight panel from a score panel (rows = ts, cols = symbol)."""
    weights = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    for t in score.index:
        book = select_book(score.loc[t], K=K, long_only=long_only)
        for sym, w in book.items():
            weights.at[t, sym] = w
    return weights


def plan_rebalance(current_weights: pd.Series, score_row: pd.Series,
                   K: int = TOP_K, long_only: bool = False,
                   min_weight_delta: float = 1e-4) -> list[RebalanceOrder]:
    """Orders to move the live book toward the target for one timestamp.

    ``current_weights`` is signed portfolio weight per symbol (0 if flat).
    Only emits an order where the target weight meaningfully differs, so an
    unchanged book costs nothing to hold (funding is earned whether or not we
    trade).
    """
    target = select_book(score_row, K=K, long_only=long_only)
    syms = set(current_weights.index) | set(target)
    orders: list[RebalanceOrder] = []
    for sym in sorted(syms):
        cur = float(current_weights.get(sym, 0.0))
        tgt = float(target.get(sym, 0.0))
        delta = tgt - cur
        if abs(delta) < min_weight_delta:
            continue
        orders.append(RebalanceOrder(symbol=sym, side="buy" if delta > 0 else "sell",
                                     target_weight=tgt, delta_weight=delta))
    return orders


def book_metrics(prices: pd.DataFrame, weights: pd.DataFrame, H: int,
                 taker_bps: float = TAKER_BPS, maker_bps: float = MAKER_BPS) -> dict:
    """Backtest a periodically-rebalanced book.

    At each rebalance bar ``p`` the target book ``weights.iloc[p]`` is held for
    ``H`` bars, so the price leg of that period is ``w_p . (C[p+H]/C[p] - 1)``
    (the H-bar simple return of the held names). Turnover is charged at each
    rebalance plus a final liquidation, entering from a flat book, so opening
    and closing positions both cost fees. Funding cashflow is added by the
    caller; this isolates the price leg.
    """
    w = weights.fillna(0.0)
    arr = w.to_numpy(float)
    px = prices[w.columns].to_numpy(float)
    n, k = arr.shape
    periods = [p for p in range(0, n, H) if p + H < n]
    gross = np.array([np.nansum(arr[p] * (px[p + H] / px[p] - 1.0)) for p in periods])

    total_turn = 0.0
    prev = np.zeros(k)
    for b in periods + [n]:                # final b == n closes to flat
        cur = arr[b] if b < n else np.zeros(k)
        total_turn += np.abs(cur - prev).sum()
        prev = cur

    out = {"n": len(periods),
           "gross_bps": float(np.nanmean(gross) * 1e4) if len(gross) else np.nan,
           "turnover": float(total_turn / max(1, len(periods))),
           "taker_bps": taker_bps, "maker_bps": maker_bps}
    for label, fee in (("taker", taker_bps), ("maker", maker_bps),
                       ("alpaca", ALPACA_TAKER_BPS)):
        cost = (total_turn / max(1, len(periods))) * fee / 1e4
        out[f"net_bps_{label}"] = out["gross_bps"] - cost * 1e4
    return out
