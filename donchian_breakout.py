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

# ── enhancements (opt-in; each is OFF at its neutral value) ────────────────────
# These are the standard Donchian/Turtle refinements. They are additive: with
# trend_filter=0, pyramid_units=1, trail_atr=0 and exit_mode="band" the code
# reduces EXACTLY to the base strategy (a test pins that).
DEFAULT_TREND_FILTER = 0    # 0 = off; else require close on the trend side of
                            # an N-bar SMA before entering (regime gate)
DEFAULT_EXIT_MODE = "band"  # "band" = opposite Donchian band; "midpoint" = the
                            # exit band's midpoint (exits sooner, keeps more)
DEFAULT_PYRAMID_UNITS = 1   # max total units (Turtle added up to 4)
DEFAULT_PYRAMID_ATR = 0.5   # add a unit every this many ATRs in the trend's favour
DEFAULT_TRAIL_ATR = 0.0     # >0 ratchets the stop to (extreme close - trail_atr*ATR)


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


def trend_mask(close, window: int) -> np.ndarray:
    """Boolean array: close is STRICTLY above its `window`-bar SMA.

    Used as a long-only regime gate — the classic false-breakout filter. The
    SMA at bar i uses closes up to and including i (available at that close),
    so gating entries on it is look-ahead-free. window<=0 -> all True (off).
    """
    c = np.asarray(close, dtype=float)
    if window <= 0:
        return np.ones(len(c), dtype=bool)
    sma = pd.Series(c).rolling(window).mean().to_numpy()
    return c > sma


def breakout_signal(high, low, close, entry: int = DEFAULT_ENTRY,
                    exit: int = DEFAULT_EXIT, allow_short: bool = False,
                    trend_filter: int = DEFAULT_TREND_FILTER,
                    exit_mode: str = DEFAULT_EXIT_MODE,
                    gate: np.ndarray | None = None) -> np.ndarray:
    """Stateful Donchian position in {-1, 0, +1} per bar.

    Long entry when close > upper(entry); long exit when close crosses the exit
    level. With `allow_short`, the mirror image is added. The exit test runs
    before the entry test, so a single bar can never both enter and exit. Flat
    until the entry band exists (first `entry` bars).

    `trend_filter` (>0): also require `close > SMA(trend_filter)` to enter long
    (and `close < SMA` to enter short) — a regime gate that suppresses
    counter-trend false breakouts. Exits are never gated.

    `exit_mode`: "band" exits when close crosses the opposite `exit` band (the
    textbook rule); "midpoint" exits when close crosses that band's MIDPOINT,
    which books profit sooner and reduces give-back on failed breakouts.

    `gate` (optional bool array): an extra regime mask ANDed with the trend
    filter for entry. None/all-True leaves the rules untouched.
    """
    c = np.asarray(close, dtype=float)
    up_e, lo_e = donchian_channel(high, low, entry)
    up_x, lo_x = donchian_channel(high, low, exit)
    if exit_mode == "midpoint":
        exit_long = (lo_x + up_x) / 2.0            # exit long below the mid
        exit_short = (lo_x + up_x) / 2.0
    else:
        exit_long, exit_short = lo_x, up_x
    ok = trend_mask(c, trend_filter)
    gated = trend_filter > 0                       # only gate entries when on
    if gate is not None:
        ok = ok & np.asarray(gate, dtype=bool)
        gated = True
    state = np.zeros(len(c), dtype=np.int8)
    pos = 0
    for i in range(len(c)):
        if np.isnan(up_e[i]) or pos == 1 and c[i] < exit_long[i] \
                or pos == -1 and c[i] > exit_short[i]:
            pos = 0
        elif pos == 0 and c[i] > up_e[i] and (ok[i] or not gated):
            pos = 1
        elif allow_short and pos == 0 and c[i] < lo_e[i] and (not ok[i] or not gated):
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
                      periods_per_year: int = 365,
                      trend_filter: int = DEFAULT_TREND_FILTER,
                      exit_mode: str = DEFAULT_EXIT_MODE,
                      pyramid_units: int = DEFAULT_PYRAMID_UNITS,
                      pyramid_atr: float = DEFAULT_PYRAMID_ATR,
                      trail_atr: float = DEFAULT_TRAIL_ATR,
                      regime: np.ndarray | None = None) -> dict:
    """Single-asset Donchian backtest with realistic per-side fees and a stop.

    `df` needs columns high/low/close. Decisions are made at each bar's CLOSE
    (the signal uses only bars up to and including that close) and the position
    then earns the NEXT bar's move — so there is no look-ahead. A bar whose LOW
    pierces the protective stop exits intra-bar at the stop price (conservative:
    assumes the stop is touched, not gapped through favourably). Fees are
    charged on the traded notional at every position change, both entry and exit.

    Enhancements (all OFF at their neutral defaults, so the base strategy is
    reproduced exactly):
      * `trend_filter` — require close on the trend side of an SMA to enter;
      * `exit_mode="midpoint"` — exit on the exit band's midpoint;
      * `pyramid_units`/`pyramid_atr` — add units as the trade moves in favour
        (Turtle add-units), so winners get a bigger position;
      * `trail_atr` — ratchet the stop behind the running extreme;
      * `regime` — optional bool array; when supplied, entries are also gated on
        it being True at that bar (all-True/None leaves the rules unchanged).

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

    sig = breakout_signal(hi, lo, o, entry, exit, allow_short,
                          trend_filter=trend_filter, exit_mode=exit_mode, gate=regime)
    a = atr(hi, lo, o, atr_window)
    cost = fee_bps / 1e4

    equity = float(initial_equity)
    eq_curve = np.empty(n)
    pos = 0                 # current direction, -1/0/+1
    units = 0.0             # absolute base-asset units held (all adds, summed)
    n_units = 0             # how many units have been added
    avg_entry = 0.0
    a_entry = np.nan
    next_add = np.nan       # price at which the next unit is added
    extreme = np.nan        # running favourable extreme for the trailing stop
    stop_price = np.nan
    entry_index = -1
    entry_equity = equity
    trades: list[dict] = []

    def _unit_size(i: int) -> float:
        u = turtle_units(o[i], a[i], risk_pct)
        return (u * equity) / o[i] if u > 0 and o[i] > 0 else 0.0

    def _open(i: int) -> None:
        nonlocal pos, units, n_units, avg_entry, a_entry, next_add, extreme
        nonlocal stop_price, entry_index, entry_equity, equity
        q = _unit_size(i)
        if q <= 0 or not np.isfinite(a[i]):
            return
        units = q
        n_units = 1
        avg_entry = o[i]
        a_entry = a[i]
        pos = 1 if sig[i] > 0 else -1
        entry_index = i
        # record equity BEFORE the entry fee so the trade's PnL includes every
        # fee it incurs (entry, adds, exit) and sum(PnL) == equity change.
        entry_equity = equity
        equity -= units * o[i] * cost                  # entry fee
        extreme = o[i]
        stop_price = avg_entry - stop_atr * a[i] if pos > 0 else avg_entry + stop_atr * a[i]
        next_add = avg_entry + pyramid_atr * a[i] if pos > 0 else avg_entry - pyramid_atr * a[i]

    def _add_unit(i: int) -> None:
        nonlocal units, n_units, avg_entry, next_add, equity
        q = _unit_size(i)
        if q <= 0:
            return
        avg_entry = (avg_entry * units + o[i] * q) / (units + q)
        units += q
        n_units += 1
        equity -= q * o[i] * cost                      # fee on the added notional
        next_add = avg_entry + pyramid_atr * a_entry if pos > 0 else avg_entry - pyramid_atr * a_entry

    def _close(i: int, price: float, reason: str) -> None:
        nonlocal pos, units, n_units, stop_price, equity
        equity -= units * price * cost                 # exit fee
        pnl = equity - entry_equity
        trades.append({
            "entry_index": entry_index, "exit_index": i,
            "side": "long" if pos > 0 else "short",
            "entry_price": avg_entry, "exit_price": price,
            "units": units, "n_units": n_units, "reason": reason, "pnl": pnl,
            "return_on_entry_equity": pnl / entry_equity if entry_equity else 0.0,
        })
        pos, units, n_units, stop_price = 0, 0.0, 0, np.nan

    for i in range(n):
        # 1) mark the position carried in from the previous close to this close.
        if pos != 0 and i > 0:
            equity += units * (o[i] - o[i - 1]) * (1.0 if pos > 0 else -1.0)
            # ratchet the trailing stop behind the running favourable extreme.
            if trail_atr > 0:
                extreme = max(extreme, o[i]) if pos > 0 else min(extreme, o[i])
                candidate = extreme - trail_atr * a[i] if pos > 0 else extreme + trail_atr * a[i]
                stop_price = max(stop_price, candidate) if pos > 0 else min(stop_price, candidate)

        # 2) protective stop against this bar's range (intra-bar).
        if pos != 0 and np.isfinite(stop_price):
            hit = lo[i] <= stop_price if pos > 0 else hi[i] >= stop_price
            if hit:
                equity += units * (stop_price - o[i - 1]) * (1.0 if pos > 0 else -1.0)
                _close(i, stop_price, "stop")

        # 3) pyramid: add a unit if price advanced pyramid_atr*A past the last add.
        if pos != 0 and n_units < pyramid_units and np.isfinite(next_add):
            reached = o[i] >= next_add if pos > 0 else o[i] <= next_add
            if reached and np.isfinite(a[i]) and a[i] > 0:
                _add_unit(i)

        # 4) act on the signal known at this close.
        want = int(sig[i])
        if want != pos:
            if pos != 0:
                _close(i, o[i], "signal")
            if want != 0:
                _open(i)

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
