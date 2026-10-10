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
#   python donchian_bot.py --paper  --source alpaca      # Alpaca daily bars
#   python donchian_bot.py --live --i-understand-the-risk
#
# Alpaca keys: this module never reads a key. The live broker imports
# portfolio/orders, which use config's trading_client built from the env vars
# APCA_API_KEY_ID / APCA_API_SECRET_KEY — the same keys as every other module.
# The Alpaca ENVIRONMENT is APCA_API_PAPER (default true = paper); the live
# banner prints which one is in effect.
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import donchian_breakout as db

_log = logging.getLogger("donchian_bot")

# Operational defaults (frozen; see docs/DONCHIAN_BREAKOUT.md).
MAX_POSITION_PCT = 0.20     # matches config.MAX_POSITION_PCT
MAX_GROSS_PCT = 1.00        # no leverage across the book
DEFAULT_EQUITY = 10_000.0

# Strategy config for the bot. BASE is the textbook rule; ENHANCED adds the
# refinements that survived the out-of-sample walk-forward in
# research/donchian_improve.py: a long-term trend gate, the channel-midpoint
# exit, and one pyramid add. On 15 symbols 2020-2026 ENHANCED beat BASE on 4/4
# test folds (the only candidate with a positive out-of-sample Sharpe) and
# improved both Sharpe (0.73 -> 0.79) and return/drawdown (3.69 -> 5.69). The
# trailing stop was tested and REJECTED (0/4 folds). `--base` reproduces the
# textbook rule. Neither is a promise; 4 folds is suggestive, not conclusive.
BASE = {}
ENHANCED = {"trend_filter": 100, "exit_mode": "midpoint", "pyramid_units": 2}

# Alpaca's universe is a SUBSET of the OKX research universe and it quotes
# pairs with a slash (BTC/USD) while OKX uses no separator (BTCUSDT). Only
# symbols present here are tradeable live; the rest are research-only and are
# skipped (never silently mapped to a symbol Alpaca might not have).
ALPACA_TRADEABLE = {"BTCUSDT": "BTC/USD", "ETHUSDT": "ETH/USD", "SOLUSDT": "SOL/USD",
                    "DOGEUSDT": "DOGE/USD", "LTCUSDT": "LTC/USD", "AVAXUSDT": "AVAX/USD",
                    "LINKUSDT": "LINK/USD", "ADAUSDT": "ADA/USD", "BCHUSDT": "BCH/USD",
                    "DOTUSDT": "DOT/USD"}


# Live crypto orders below Alpaca's minimum notional are rejected by the venue;
# skipping them locally avoids a guaranteed-failing round-trip every cycle. Alpaca
# documents a $1 minimum for crypto; default a touch above it.
MIN_ORDER_NOTIONAL = float(os.getenv("DONCHIAN_MIN_NOTIONAL", "1.0"))


def to_alpaca_symbol(sym: str) -> str | None:
    return ALPACA_TRADEABLE.get(sym)


def from_alpaca_symbol(sym: str) -> str | None:
    return next((k for k, v in ALPACA_TRADEABLE.items() if v == sym), None)


# ── data sources: OKX (research cache) or Alpaca (live decisions) ──────────────
OKX_DAILY_URL = "https://www.okx.com/api/v5/market/history-candles"
ALPACA_BARS_URL = "https://data.alpaca.markets/v1beta3/crypto/us/bars"
ALPACA_KEYS = ("open", "high", "low", "close", "volume")


def _http_json(url: str, headers: dict | None = None) -> dict:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def fetch_okx_daily(symbols, days: int = 2200, inst_suffix: str = "-USDT") -> dict:
    """Daily OHLCV from OKX's public history-candles endpoint (no auth).

    `symbols` are research names (BTCUSDT); the OKX instId is BTC-USDT. Paginates
    with `after` on the oldest open_time. Returns {symbol: DataFrame(high/low/close)}
    indexed by UTC timestamp — the shape the backtest and the bot both expect.
    """
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        inst = sym.replace("USDT", inst_suffix) if inst_suffix else sym
        rows, after = [], None
        while True:
            q = {"instId": inst, "bar": "1D", "limit": "300"}
            if after:
                q["after"] = str(after)
            payload = _http_json(f"{OKX_DAILY_URL}?{urllib.parse.urlencode(q)}")
            batch = payload.get("data", [])
            if not batch:
                break
            rows.extend(batch)
            after = batch[-1][0]                       # open_time (ms, newest last)
            if len(rows) >= days + 5:
                break
        if not rows:
            continue
        recs = []
        for r in rows:
            ts = int(r[0])
            if ts > 10**14:                            # OKX switched ms->us mid-history
                ts //= 1000
            recs.append((ts, float(r[2]), float(r[3]), float(r[4])))  # ts, high, low, close
        df = pd.DataFrame(recs, columns=["ts", "high", "low", "close"])
        df = df.assign(ts=pd.to_datetime(df["ts"], unit="ms", utc=True))
        df = df.set_index("ts").sort_index()
        df = df[~df.index.duplicated()].tail(days)
        if len(df) > 200:
            out[sym] = df
    return out


def _alpaca_bars(pair: str, start: str, key: str, secret: str) -> list:
    """Prefer alpaca-py's client (shared keys); fall back to REST, paginating
    with next_page_token."""
    try:
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame
        client = CryptoHistoricalDataClient(api_key=key, secret_key=secret)
        bars = client.get_crypto_bars(CryptoBarsRequest(
            symbol_or_symbols=pair, timeframe=TimeFrame.Day, start=start)).data.get(pair, [])
        return [{"t": b.timestamp, "open": b.open, "high": b.high, "low": b.low,
                 "close": b.close, "volume": b.volume} for b in bars]
    except Exception as e:  # noqa: BLE001 — any alpaca-py import/call failure -> REST
        _log.debug("alpaca-py bars unavailable, using REST: %s", e)
    out, token = [], None
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    while True:
        q = {"symbols": pair, "timeframe": "1Day", "start": start, "limit": "1000"}
        if token:
            q["page_token"] = token
        payload = _http_json(f"{ALPACA_BARS_URL}?{urllib.parse.urlencode(q)}", headers)
        out.extend(payload.get("bars", {}).get(pair, []))
        token = payload.get("next_page_token")
        if not token:
            break
    return out


def fetch_alpaca_daily(symbols, days: int = 2200, key: str | None = None,
                       secret: str | None = None) -> dict:
    """Daily OHLCV bars from Alpaca's crypto data API.

    Tries alpaca-py first (same `CryptoHistoricalDataClient` config.py builds, so
    the SAME keys are used), and falls back to the REST endpoint with an
    APCA-API-KEY-ID/APCA-API-SECRET-KEY header. Returns {research_symbol: frame}.
    Requires real keys; raises if none are configured.
    """
    key = key or os.getenv("APCA_API_KEY_ID")
    secret = secret or os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Alpaca keys missing: set APCA_API_KEY_ID / APCA_API_SECRET_KEY")
    start = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00Z")
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        pair = to_alpaca_symbol(sym) or sym
        rows = _alpaca_bars(pair, start, key, secret)
        if not rows:
            continue
        df = pd.DataFrame([{k: float(b.get(k) or 0) for k in ALPACA_KEYS} |
                           {"ts": pd.Timestamp(b["t"])} for b in rows])
        df = df.set_index("ts").sort_index()
        df = df[~df.index.duplicated()].tail(days)[["high", "low", "close"]]
        if len(df) > 200:
            out[sym] = df
    return out


def default_universe() -> list[str]:
    """Symbols with a live signal to evaluate: the Alpaca-tradeable set unless
    DONCHIAN_UNIVERSE overrides it (comma-separated research names)."""
    override = os.getenv("DONCHIAN_UNIVERSE")
    if override:
        return [s.strip() for s in override.split(",") if s.strip()]
    return sorted(ALPACA_TRADEABLE)


def alpaca_env() -> str:
    """Which Alpaca environment the live client will hit: 'paper' or 'live'.

    Mirrors config.PAPER (`APCA_API_PAPER`, default true) WITHOUT importing
    config, so the banner never needs real keys. Defaults to 'paper' — the safe,
    user-chosen target."""
    return "paper" if os.getenv("APCA_API_PAPER", "true").lower() == "true" else "live"


def _cache_universe(cache: str) -> list[str]:
    """Symbols present in a close-cache dir that Alpaca can actually trade.

    A cached paper book must not contain legs live could never place, so the
    OKX research cache (15 names) is trimmed to the Alpaca-tradeable subset."""
    try:
        from research.xsec_momentum import load_close
        cols = list(load_close(cache).columns)
    except Exception:  # noqa: BLE001 — a missing/odd cache just means "no filter"
        return []
    return [s for s in cols if s in ALPACA_TRADEABLE]


def write_close_cache(data: dict, cache: str) -> int:
    """Persist fetched bars as a close cache (the shape load_close reads) so the
    forward report marks the ledger on the SAME prices the bot decided on.

    Writes {symbol}_1D.csv with open_time(ms) + close. Returns symbols written."""
    p = Path(cache)
    p.mkdir(parents=True, exist_ok=True)
    n = 0
    for sym, df in data.items():
        if "close" not in df.columns:
            continue
        # .asi8 is in the index's OWN unit (ms for venue bars), so convert to ns
        # first — otherwise //10**6 silently truncates the epoch to ~7 digits.
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        open_time_ms = idx.astype("datetime64[ns]").astype("int64") // 10**6
        pd.DataFrame({
            "open_time": open_time_ms,
            "close": df["close"].to_numpy(),
        }).to_csv(p / f"{sym}_1D.csv", index=False)
        n += 1
    return n


def load_source(source: str, cache: str = "okx_daily", days: int = 2200) -> dict:
    """Load daily bars for the bot's universe from OKX or Alpaca.

    Alpaca is the honest choice for this deployment: it is where orders settle
    and, unlike OKX's public API, it is not geo-blocked. The OKX path remains for
    reproducible research and falls back to the on-disk cache when the network is
    unavailable. Both return {symbol: DataFrame(high/low/close)}.
    """
    if source == "okx":
        try:
            data = fetch_okx_daily(default_universe(), days=days)
        except Exception as e:  # noqa: BLE001 — fall back to cache on any fetch error
            _log.warning("OKX daily fetch failed (%s); falling back to %s", e, cache)
            data = {}
        if data:
            return data
        # fall back to the on-disk research cache if the live fetch is unavailable
        from research.donchian_backtest import load_daily
        data = load_daily(cache)
        keep = set(_cache_universe(cache))
        if keep:
            data = {k: v for k, v in data.items() if k in keep}
        return data
    if source == "alpaca":
        return fetch_alpaca_daily(default_universe(), days=days)
    raise ValueError(f"unknown source {source!r} (use 'okx' or 'alpaca')")


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


# breakout_signal()'s own keywords; anything else in a strategy dict is an
# execution lever handled here (pyramiding), not by the signal.
_SIGNAL_KEYS = ("trend_filter", "exit_mode", "allow_short")


def _split_strategy(strategy: dict | None) -> tuple[dict, dict]:
    s = strategy or {}
    return {k: v for k, v in s.items() if k in _SIGNAL_KEYS}, \
           {k: v for k, v in s.items() if k not in _SIGNAL_KEYS}


def units_for_trend(price: float, band: float, atr_value: float,
                    max_units: int = db.DEFAULT_PYRAMID_UNITS,
                    step_atr: float = db.DEFAULT_PYRAMID_ATR) -> int:
    """How many Turtle units a live long should hold, given how far the price has
    run ABOVE the entry band (stateless, causal — uses only the current bar).

    One unit at the breakout; one more for every `step_atr` ATRs of progress,
    capped at `max_units`. This mirrors the backtest's pyramid adds without
    needing to remember the entry price across cycles.
    """
    if max_units <= 1 or not np.isfinite(atr_value) or atr_value <= 0:
        return max(1, int(max_units))
    progress = (price - band) / (atr_value * step_atr)
    return int(min(max_units, 1 + max(0, np.floor(progress))))


def latest_signal(df: pd.DataFrame, strategy: dict | None = None):
    """Return (signal, price, atr, entry_band) at the most recent close. `df`
    needs high/low/close and at least entry+1 rows. `strategy` is a kwargs-style
    dict (e.g. donchian_bot.ENHANCED); signal-only keys are passed through."""
    sig_kw, _ = _split_strategy(strategy)
    sig = db.breakout_signal(df["high"], df["low"], df["close"],
                             db.DEFAULT_ENTRY, db.DEFAULT_EXIT, **sig_kw)
    a = db.atr(df["high"], df["low"], df["close"], db.DEFAULT_ATR)
    up, _ = db.donchian_channel(df["high"], df["low"], db.DEFAULT_ENTRY)
    return int(sig[-1]), float(df["close"].iloc[-1]), float(a[-1]), float(up[-1])


def target_book(data: dict[str, pd.DataFrame], equity: float,
                risk_pct: float = db.DEFAULT_RISK_PCT,
                max_position_pct: float = MAX_POSITION_PCT,
                max_gross_pct: float = MAX_GROSS_PCT,
                strategy: dict | None = None) -> dict[str, Target]:
    """Target signed notional weights for every symbol with a live signal.

    Each active symbol is sized by turtle_units (a 1-ATR move costs risk_pct),
    scaled by how many Turtle units the current trend warrants (pyramiding),
    capped at max_position_pct, then the whole book is scaled down so gross
    exposure never exceeds max_gross_pct. Long-only by default.
    """
    _, exec_kw = _split_strategy(strategy)
    max_units = int(exec_kw.get("pyramid_units", db.DEFAULT_PYRAMID_UNITS))
    step_atr = float(exec_kw.get("pyramid_atr", db.DEFAULT_PYRAMID_ATR))
    raw: dict[str, Target] = {}
    for sym, df in data.items():
        if len(df) < db.DEFAULT_ENTRY + 2:
            continue
        sig, price, a, band = latest_signal(df, strategy)
        if sig == 0 or not np.isfinite(a):
            continue
        units = units_for_trend(price, band, a, max_units, step_atr) if sig > 0 else 1
        w = db.turtle_units(price, a, risk_pct, max_notional_pct=max_position_pct) * units
        w = min(w, max_position_pct)
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
LEDGER_COLS = ["date", "equity", "mode", "book", "gross_weight", "n_legs"]


def append_ledger(path: str, date: str, equity: float, book: dict[str, Target],
                  mode: str = "base") -> bool:
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
        "date": date, "equity": f"{equity:.2f}", "mode": mode,
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
            # Account equity (cash + open positions), NOT buying power: for
            # crypto/margin accounts buying power can be margin-inflated or read
            # 0.0, which would mis-size every order. See portfolio.get_account_equity.
            return float(portfolio.get_account_equity())

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
              ledger: str | None = None, live: bool = False,
              strategy: dict | None = None, mode: str = "base") -> list[Decision]:
    """One rebalance cycle: read broker state, plan, and act. A plain function so
    it can be tested with a fake broker and no network."""
    equity = broker.equity()
    prices = {s: float(df["close"].iloc[-1]) for s, df in data.items()}
    targets = target_book(data, equity, strategy=strategy)
    current = current_weights(broker.positions(), prices, equity)
    decisions = plan_rebalance(targets, current)
    if ledger:
        append_ledger(ledger, date, equity, targets, mode=mode)
    if live:
        # Only symbols Alpaca actually lists may be traded live; research-only
        # symbols are logged in the ledger but never ordered.
        for d in decisions:
            if d.side in ("buy", "sell") and np.isfinite(d.price) and to_alpaca_symbol(d.symbol):
                notional = abs(d.delta_weight) * equity
                # Venue rejects sub-minimum crypto orders; skip before the
                # round-trip rather than submitting a known-failing order.
                if notional < MIN_ORDER_NOTIONAL:
                    _log.info("skip %s %s: notional %.4f < min %.4f",
                              d.side, d.symbol, notional, MIN_ORDER_NOTIONAL)
                    continue
                qty = broker.quantity_for_notional(d.symbol, notional, d.price)
                if qty > 0:
                    broker.submit(d.symbol, d.side, qty)
    return decisions


def main(argv=None):
    ap = argparse.ArgumentParser(description="Daily Donchian breakout bot")
    ap.add_argument("--cache", default="okx_daily")
    ap.add_argument("--source", choices=("okx", "alpaca"), default="okx",
                    help="where decisions come from: okx (research, default) or "
                         "alpaca (live bars — the venue you actually trade on)")
    ap.add_argument("--days", type=int, default=2200,
                    help="history depth to fetch for --source network loads")
    ap.add_argument("--paper", action="store_true", help="paper ledger; nothing traded")
    ap.add_argument("--live", action="store_true", help="submit real orders")
    ap.add_argument("--i-understand-the-risk", action="store_true", dest="ack")
    ap.add_argument("--ledger", default="donchian_ledger.csv")
    ap.add_argument("--save-cache", metavar="DIR", default=None,
                    help="persist the fetched bars as a close cache so the forward "
                         "report marks the ledger on the SAME prices")
    ap.add_argument("--equity", type=float, default=DEFAULT_EQUITY)
    ap.add_argument("--base", action="store_true",
                    help="use the textbook rule instead of the enhanced (trend gate + pyramid) default")
    a = ap.parse_args(argv)

    if a.live and not a.ack:
        print("refused: --live requires --i-understand-the-risk (see docs/DONCHIAN_BREAKOUT.md)")
        return 2
    if not a.live and not a.paper:
        print("nothing to do: pass --paper (default safe) or --live --i-understand-the-risk")
        return 2

    source = a.source
    try:
        data = load_source(source, cache=a.cache, days=a.days)
    except Exception as e:  # noqa: BLE001 — surface any load failure as exit 1
        if source == "alpaca":
            # no keys / API unreachable: degrade to the venue-accurate research
            # cache rather than fail the whole cron run.
            print(f"alpaca bars unavailable ({e}); falling back to {a.cache}")
            source = "okx"
            data = load_source("okx", cache=a.cache, days=a.days)
        else:
            print(f"failed to load {source} data: {e}")
            return 1
    if not data:
        print(f"no data from source={a.source}"
              + (f" (cache {a.cache}/ empty — run research/fetch_okx_daily.py)" if source == "okx" else ""))
        return 1
    if a.save_cache:
        n = write_close_cache(data, a.save_cache)
        print(f"saved {n} symbols to close cache {a.save_cache}/")
    strategy = BASE if a.base else ENHANCED
    mode = "base" if a.base else "enhanced"
    date = max(df.index.max() for df in data.values()).date().isoformat()
    broker = make_alpaca_broker() if a.live else PaperBroker(a.equity)
    decisions = run_cycle(data, broker, date, ledger=a.ledger if a.paper else None,
                          live=a.live, strategy=strategy, mode=mode)
    env = f"  alpaca_env={alpaca_env()}" if a.live else ""
    print(f"{date}  {'LIVE' if a.live else 'PAPER'}  source={source}  mode={mode}  "
          f"equity={broker.equity():.2f}  symbols={len(data)}{env}")
    for d in decisions:
        if d.side != "hold":
            print(f"  {d.side:<4} {d.symbol:<9} Δw={d.delta_weight:+.3f} "
                  f"target={d.target_weight:+.3f} ({d.reason})")
    if a.paper:
        print(f"  ledger row written to {a.ledger} (paper only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
