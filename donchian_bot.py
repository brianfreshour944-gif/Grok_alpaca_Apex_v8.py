# donchian_bot.py — daily Donchian-breakout bot (paper-first, pluggable broker).
#
# Separate, additive entry point. It does NOT touch config.py or main_bot.py.
# The strategy lives in donchian_breakout.py; this module is the operational
# shell: read daily bars -> target book -> reconcile to holdings -> (paper) log
# or (live) submit orders through a broker adapter.
#
# Live trading is opt-in and double-gated: `--live` alone is refused; you must
# also pass --i-understand-the-risk. Default is a PAPER ledger (nothing traded).
#
# Usage:
#   python donchian_bot.py --paper  --ledger donchian_ledger.csv
#   python donchian_bot.py --live --i-understand-the-risk
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import donchian_breakout as db

# Operational defaults (frozen; see docs/DONCHIAN_BREAKOUT.md).
MAX_POSITION_PCT = 0.20     # matches config.MAX_POSITION_PCT
MAX_GROSS_PCT = 1.00        # no leverage across the book
DEFAULT_EQUITY = 10_000.0

# Alpaca's universe is a SUBSET of the OKX research universe and it quotes
# pairs with a slash (BTC/USD) while OKX uses no separator (BTCUSDT). Only
# symbols present here are tradeable live; the rest are research-only and are
# skipped (never silently mapped to a symbol Alpaca might not have).
ALPACA_TRADEABLE = {"BTCUSDT": "BTC/USD", "ETHUSDT": "ETH/USD", "SOLUSDT": "SOL/USD",
                    "DOGEUSDT": "DOGE/USD", "LTCUSDT": "LTC/USD", "AVAXUSDT": "AVAX/USD",
                    "LINKUSDT": "LINK/USD", "ADAUSDT": "ADA/USD", "BCHUSDT": "BCH/USD",
                    "DOTUSDT": "DOT/USD"}


def to_alpaca_symbol(sym: str) -> str | None:
    return ALPACA_TRADEABLE.get(sym)


def from_alpaca_symbol(sym: str) -> str | None:
    return next((k for k, v in ALPACA_TRADEABLE.items() if v == sym), None)


@dataclass
class Target:
    symbol: str
    signal: int                 # -1 / 0 / +1
    price: float
    atr: float
    weight: float               # signed target notional / equity


@dataclass
class Decision:
    symbol: str
    side: str                   # "buy" / "sell" / "hold"
    delta_weight: float
    target_weight: float
    current_weight: float
    price: float
    reason: str


def latest_signal(df: pd.DataFrame) -> tuple[int, float, float]:
    """Return (signal, price, atr) at the most recent close. `df` needs
    high/low/close and at least entry+1 rows."""
    sig = db.breakout_signal(df["high"], df["low"], df["close"],
                             db.DEFAULT_ENTRY, db.DEFAULT_EXIT)
    a = db.atr(df["high"], df["low"], df["close"], db.DEFAULT_ATR)
    return int(sig[-1]), float(df["close"].iloc[-1]), float(a[-1])


def target_book(data: dict[str, pd.DataFrame], equity: float,
                risk_pct: float = db.DEFAULT_RISK_PCT,
                max_position_pct: float = MAX_POSITION_PCT,
                max_gross_pct: float = MAX_GROSS_PCT) -> dict[str, Target]:
    """Target signed notional weights for every symbol with a live signal.

    Each active symbol is sized by turtle_units (a 1-ATR move costs risk_pct),
    capped at max_position_pct, then the whole book is scaled down so gross
    exposure never exceeds max_gross_pct. Long-only by default.
    """
    raw: dict[str, Target] = {}
    for sym, df in data.items():
        if len(df) < db.DEFAULT_ENTRY + 2:
            continue
        sig, price, a = latest_signal(df)
        if sig == 0 or not np.isfinite(a):
            continue
        w = db.turtle_units(price, a, risk_pct, max_notional_pct=max_position_pct)
        if w > 0:
            raw[sym] = Target(sym, sig, price, a, sig * w)
    gross = sum(abs(t.weight) for t in raw.values())
    if gross > max_gross_pct and gross > 0:
        scale = max_gross_pct / gross
        for t in raw.values():
            t.weight *= scale
    return raw


def current_weights(positions: dict[str, float], prices: dict[str, float],
                    equity: float) -> dict[str, float]:
    """Signed current notional / equity from {symbol: qty} and {symbol: price}."""
    return {s: (q * prices.get(s, 0.0)) / equity for s, q in positions.items() if equity}


def plan_rebalance(targets: dict[str, Target], current: dict[str, float],
                   min_trade_weight: float = 0.005) -> list[Decision]:
    """Diff target vs current weights into buy/sell/hold decisions.

    `min_trade_weight` suppresses dust rebalances. Symbols no longer targeted
    (signal flat, or dropped from the universe) are sold to zero.
    """
    out: list[Decision] = []
    for sym in sorted(set(targets) | set(current)):
        tgt = targets.get(sym)
        tw = tgt.weight if tgt else 0.0
        cw = current.get(sym, 0.0)
        delta = tw - cw
        price = tgt.price if tgt else float("nan")
        if abs(delta) < min_trade_weight:
            out.append(Decision(sym, "hold", 0.0, tw, cw, price, "within band"))
            continue
        side = "buy" if delta > 0 else "sell"
        reason = "enter" if tgt and cw == 0 else "exit" if not tgt else "resize"
        out.append(Decision(sym, side, delta, tw, cw, price, reason))
    return out


# ── forward ledger ─────────────────────────────────────────────────────────────
LEDGER_COLS = ["date", "equity", "book", "gross_weight", "n_legs"]


def append_ledger(path: str, date: str, equity: float, book: dict[str, Target]) -> bool:
    """Append one hypothetical daily book row, idempotent by date. Returns True
    if a row was written (False if the date already existed)."""
    p = Path(path)
    rows = []
    if p.exists():
        with p.open() as f:
            rows = list(csv.DictReader(f))
    if any(r["date"] == date for r in rows):
        return False
    rows.append({
        "date": date, "equity": f"{equity:.2f}",
        "book": "|".join(f"{s}:{t.signal}:{t.weight:+.3f}" for s, t in sorted(book.items())),
        "gross_weight": f"{sum(abs(t.weight) for t in book.values()):.3f}",
        "n_legs": str(len(book)),
    })
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLS)
        w.writeheader()
        w.writerows(rows)
    return True


# ── broker adapters ────────────────────────────────────────────────────────────
class PaperBroker:
    """No-op broker: records intended orders, trades nothing."""

    def __init__(self, equity: float = DEFAULT_EQUITY):
        self._equity = equity
        self.placed: list[dict] = []

    def equity(self) -> float:
        return self._equity

    def positions(self) -> dict[str, float]:
        return {}

    def quantity_for_notional(self, symbol: str, notional: float, price: float) -> float:
        return notional / price if price else 0.0

    def submit(self, symbol: str, side: str, qty: float) -> str:
        self.placed.append({"symbol": symbol, "side": side, "qty": qty})
        return f"paper-{len(self.placed)}"


def make_alpaca_broker():
    """Lazily build a live Alpaca broker adapter. Imports config/orders/portfolio
    only when called, so the module stays import-safe and dependency-isolated."""
    import asyncio

    from alpaca.trading.enums import OrderSide

    import orders
    import portfolio

    class AlpacaBroker:
        def equity(self) -> float:
            return float(portfolio.get_buying_power())

        def positions(self) -> dict[str, float]:
            # return keys in the RESEARCH convention (BTCUSDT) so plan_rebalance
            # compares like for like; unmapped Alpaca symbols are ignored.
            out = {}
            for s, p in portfolio.get_all_positions().items():
                key = from_alpaca_symbol(s)
                if key:
                    out[key] = float(p.get("qty", 0))
            return out

        def quantity_for_notional(self, symbol: str, notional: float, price: float) -> float:
            return notional / price if price else 0.0

        def submit(self, symbol: str, side: str, qty: float) -> str:
            alpaca_sym = to_alpaca_symbol(symbol) or symbol
            oid: dict = {}
            asyncio.run(orders.place_order(
                alpaca_sym, OrderSide.BUY if side == "buy" else OrderSide.SELL,
                qty, market=True, order_id_out=oid))
            return str(oid.get("order_id"))

    return AlpacaBroker()


def run_cycle(data: dict[str, pd.DataFrame], broker, date: str,
              ledger: str | None = None, live: bool = False) -> list[Decision]:
    """One rebalance cycle: read broker state, plan, and act. A plain function so
    it can be tested with a fake broker and no network."""
    equity = broker.equity()
    prices = {s: float(df["close"].iloc[-1]) for s, df in data.items()}
    targets = target_book(data, equity)
    current = current_weights(broker.positions(), prices, equity)
    decisions = plan_rebalance(targets, current)
    if ledger:
        append_ledger(ledger, date, equity, targets)
    if live:
        # Only symbols Alpaca actually lists may be traded live; research-only
        # symbols are logged in the ledger but never ordered.
        for d in decisions:
            if d.side in ("buy", "sell") and np.isfinite(d.price) and to_alpaca_symbol(d.symbol):
                notional = abs(d.delta_weight) * equity
                qty = broker.quantity_for_notional(d.symbol, notional, d.price)
                if qty > 0:
                    broker.submit(d.symbol, d.side, qty)
    return decisions


def main(argv=None):
    ap = argparse.ArgumentParser(description="Daily Donchian breakout bot")
    ap.add_argument("--cache", default="okx_daily")
    ap.add_argument("--paper", action="store_true", help="paper ledger; nothing traded")
    ap.add_argument("--live", action="store_true", help="submit real orders")
    ap.add_argument("--i-understand-the-risk", action="store_true", dest="ack")
    ap.add_argument("--ledger", default="donchian_ledger.csv")
    ap.add_argument("--equity", type=float, default=DEFAULT_EQUITY)
    a = ap.parse_args(argv)

    if a.live and not a.ack:
        print("refused: --live requires --i-understand-the-risk (see docs/DONCHIAN_BREAKOUT.md)")
        return 2
    if not a.live and not a.paper:
        print("nothing to do: pass --paper (default safe) or --live --i-understand-the-risk")
        return 2

    from research.donchian_backtest import load_daily
    data = load_daily(a.cache)
    if not data:
        print(f"no data in {a.cache}/ — run research/fetch_okx_daily.py first")
        return 1
    date = max(df.index.max() for df in data.values()).date().isoformat()
    broker = make_alpaca_broker() if a.live else PaperBroker(a.equity)
    decisions = run_cycle(data, broker, date,
                          ledger=a.ledger if a.paper else None, live=a.live)
    print(f"{date}  {'LIVE' if a.live else 'PAPER'}  equity={broker.equity():.2f}")
    for d in decisions:
        if d.side != "hold":
            print(f"  {d.side:<4} {d.symbol:<9} Δw={d.delta_weight:+.3f} "
                  f"target={d.target_weight:+.3f} ({d.reason})")
    if a.paper:
        print(f"  ledger row written to {a.ledger} (paper only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
