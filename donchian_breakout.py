# donchian_breakout.py — Donchian-channel (Turtle-style) breakout strategy.
#
# Pure, dependency-light (numpy/pandas only). Deliberately does NOT import
# config/alpaca, so it is importable and unit-testable in isolation and can be
# dropped into the live bot without dragging the Alpaca clients in.
#
# The classic rules (Turtle / Donchian):
#   ENTRY : close breaks above the highest HIGH of the prior `entry` bars
#           (or below the lowest LOW for the short side).
#   EXIT  : close crosses the opposite band over a shorter `exit` window.
#   SIZE  : one "unit" is sized so a 1-ATR adverse move costs `risk_pct` of
#           equity (the Turtle N-unit rule).
#   STOP  : a protective stop `stop_atr` ATRs behind the entry.
#
# No look-ahead: the band at bar t is built from bars [t-window, t-1] only, so
# the close at t is compared against a band that does not contain t. See
# docs/DONCHIAN_BREAKOUT.md and tests/test_donchian_breakout.py.
from __future__ import annotations

import numpy as np
import pandas as pd

# Frozen defaults. These are the textbook Donchian parameters, not tuned on the
# backtest sample; research/donchian_backtest.py sweeps them only to show how
# fragile a tuned peak would be. Do not treat a swept value as a promotion.
DEFAULT_ENTRY = 20
DEFAULT_EXIT = 10
DEFAULT_ATR = 20
DEFAULT_RISK_PCT = 0.01
DEFAULT_FEE_BPS = 25.0     # Alpaca crypto taker, per side (config.ESTIMATED_TAKER_FEE_BPS)
DEFAULT_STOP_ATR = 2.0


def donchian_channel(high, low, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling Donchian band from the `window` bars *before* each bar.

    Returns (upper, lower). Entry i is NaN for the first `window` bars, then
    upper[i] = max(high[i-window : i]) and lower[i] = min(low[i-window : i]) —
    the current bar is EXCLUDED, which is what makes the breakout test
    look-ahead-free.
    """
    h = pd.Series(np.asarray(high, dtype=float))
    l = pd.Series(np.asarray(low, dtype=float))
    upper = h.rolling(window).max().shift(1).to_numpy()
    lower = l.rolling(window).min().shift(1).to_numpy()
    return upper, lower


def atr(high, low, close, window: int = DEFAULT_ATR) -> np.ndarray:
    """Average true range over `window` bars (Wilder's smoothing is overkill
    here; a simple mean matches the original Turtle N closely enough)."""
    h, l, c = (np.asarray(x, dtype=float) for x in (high, low, close))
    prev_c = np.concatenate([[np.nan], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    return pd.Series(tr).rolling(window).mean().to_numpy()


def breakout_signal(high, low, close, entry: int = DEFAULT_ENTRY,
                    exit: int = DEFAULT_EXIT, allow_short: bool = False) -> np.ndarray:
    """Stateful Donchian position in {-1, 0, +1} per bar.

    Long entry when close > upper(entry); long exit when close < lower(exit).
    With `allow_short`, the mirror image is added: short entry when
    close < lower(entry); short exit when close > upper(exit). The exit test is
    evaluated before the entry test each bar, so a single bar can never both
    enter and exit. Flat until the entry band exists (first `entry` bars).
    """
    c = np.asarray(close, dtype=float)
    up_e, lo_e = donchian_channel(high, low, entry)
    up_x, lo_x = donchian_channel(high, low, exit)
    state = np.zeros(len(c), dtype=np.int8)
    pos = 0
    for i in range(len(c)):
        if np.isnan(up_e[i]) or pos == 1 and c[i] < lo_x[i] or pos == -1 and c[i] > up_x[i]:
            pos = 0
        elif pos == 0 and c[i] > up_e[i]:
            pos = 1
        elif allow_short and pos == 0 and c[i] < lo_e[i]:
            pos = -1
        state[i] = pos
    return state


def turtle_units(price: float, atr_value: float, risk_pct: float = DEFAULT_RISK_PCT,
                 max_notional_pct: float = 1.0) -> float:
    """Notional of one unit as a FRACTION of equity.

    Sized so a 1-ATR adverse move equals `risk_pct` of equity:
        unit_notional = risk_pct * equity * price / ATR   =>   fraction = risk_pct * price / ATR
    Capped at `max_notional_pct` (and at 0 for a non-positive/degenerate ATR)
    so a low-volatility reading cannot lever the book to infinity.
    """
    if not np.isfinite(price) or not np.isfinite(atr_value) or price <= 0 or atr_value <= 0:
        return 0.0
    return float(min(risk_pct * price / atr_value, max_notional_pct))


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))


def _sharpe(returns: np.ndarray, periods_per_year: int) -> float:
    r = returns[np.isfinite(returns)]
    if r.size < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def backtest_donchian(df: pd.DataFrame, entry: int = DEFAULT_ENTRY, exit: int = DEFAULT_EXIT,
                      atr_window: int = DEFAULT_ATR, risk_pct: float = DEFAULT_RISK_PCT,
                      fee_bps: float = DEFAULT_FEE_BPS, stop_atr: float = DEFAULT_STOP_ATR,
                      allow_short: bool = False, initial_equity: float = 10_000.0,
                      periods_per_year: int = 365) -> dict:
    """Single-asset Donchian backtest with realistic per-side fees and a stop.

    `df` needs columns high/low/close. Decisions are made at each bar's CLOSE
    (the signal uses only bars up to and including that close) and the position
    then earns the NEXT bar's move — so there is no look-ahead. A bar whose LOW
    pierces the protective stop exits intra-bar at the stop price (conservative:
    assumes the stop is touched, not gapped through favourably). Fees are
    charged on the traded notional at every position change, both entry and exit.

    Equity is the single source of truth: a trade's PnL is the equity change
    between opening and closing, so fees and marks can never drift out of sync.
    Returns a dict with the equity curve, per-trade records and summary metrics.
    `exposure` is the fraction of bars holding a position.
    """
    for col in ("high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"df missing required column {col!r}")

    o = df["close"].to_numpy(float)          # decisions/fills at the close
    hi = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    n = len(df)
    if n < max(entry, exit, atr_window) + 2:
        raise ValueError("not enough bars for the requested windows")

    sig = breakout_signal(hi, lo, o, entry, exit, allow_short)
    a = atr(hi, lo, o, atr_window)
    cost = fee_bps / 1e4

    equity = float(initial_equity)
    eq_curve = np.empty(n)
    pos = 0                 # current direction, -1/0/+1
    units = 0.0             # absolute units held
    entry_price = 0.0
    stop_price = np.nan
    entry_index = -1
    entry_equity = equity
    trades: list[dict] = []

    def _open(i: int, want: int) -> None:
        nonlocal pos, units, entry_price, stop_price, entry_equity, entry_index, equity
        u = turtle_units(o[i], a[i], risk_pct)
        if u <= 0:
            return
        notional = u * equity
        units = notional / o[i]
        entry_price = o[i]
        equity -= notional * cost                      # entry fee
        pos = want
        entry_index = i
        entry_equity = equity
        stop_price = entry_price - stop_atr * a[i] if pos > 0 else entry_price + stop_atr * a[i]

    def _close(i: int, price: float, reason: str) -> None:
        nonlocal pos, units, stop_price, equity
        equity -= units * price * cost                 # exit fee
        pnl = equity - entry_equity
        trades.append({
            "entry_index": entry_index, "exit_index": i,
            "side": "long" if pos > 0 else "short",
            "entry_price": entry_price, "exit_price": price,
            "units": units, "reason": reason, "pnl": pnl,
            "return_on_entry_equity": pnl / entry_equity if entry_equity else 0.0,
        })
        pos, units, stop_price = 0, 0.0, np.nan

    for i in range(n):
        # 1) mark the position carried in from the previous close to this close.
        if pos != 0 and i > 0:
            equity += units * (o[i] - o[i - 1]) * (1.0 if pos > 0 else -1.0)

        # 2) protective stop against this bar's range (intra-bar).
        if pos != 0 and np.isfinite(stop_price):
            hit = lo[i] <= stop_price if pos > 0 else hi[i] >= stop_price
            if hit:
                equity += units * (stop_price - o[i - 1]) * (1.0 if pos > 0 else -1.0)
                _close(i, stop_price, "stop")

        # 3) act on the signal known at this close.
        want = int(sig[i])
        if want != pos:
            if pos != 0:
                _close(i, o[i], "signal")
            if want != 0:
                _open(i, want)

        eq_curve[i] = equity

    # Liquidate any open position at the final close so the equity curve and the
    # sum of trade PnL reconcile exactly (an open mark would otherwise hide in
    # the last equity point without a matching trade record).
    if pos != 0:
        _close(n - 1, o[n - 1], "eod")
        eq_curve[n - 1] = equity

    eq = pd.Series(eq_curve, index=df.index)
    daily = eq.pct_change().fillna(0.0).to_numpy()
    tr = pd.DataFrame(trades)
    wins = tr[tr["pnl"] > 0]["pnl"] if not tr.empty else pd.Series(dtype=float)
    losses = tr[tr["pnl"] <= 0]["pnl"] if not tr.empty else pd.Series(dtype=float)
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    return {
        "equity": eq,
        "trades": tr,
        "initial_equity": initial_equity,
        "final_equity": float(equity),
        "total_return": float(equity / initial_equity - 1.0),
        "cagr": float((equity / initial_equity) ** (periods_per_year / max(n, 1)) - 1.0),
        "sharpe": _sharpe(daily, periods_per_year),
        "max_drawdown": _max_drawdown(eq_curve),
        "n_trades": len(tr),
        "win_rate": float((tr["pnl"] > 0).mean()) if not tr.empty else 0.0,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0,
        "exposure": float((sig != 0).mean()),
        "buy_hold_return": float(o[-1] / o[0] - 1.0),
        "signal": sig,
    }
