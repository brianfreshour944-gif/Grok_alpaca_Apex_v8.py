# tests/test_protective_exits.py — structural guards that main_bot.py's
# protective-exit paths submit MARKET sells for the quantity actually held.
#
# The exit DECISION logic is covered behaviorally in test_exit_logic.py, and
# place_order(market=True) is covered in test_orders.py. What is not otherwise
# reachable without standing up the whole async trading loop is the WIRING in
# main_bot.py: that the stop-loss / slow-bleed / max-hold branch and the
# kill-switch branch pass market=True (rather than a resting price*0.999
# limit) and sell the live position qty. These assert that contract from the
# source so a future edit can't silently regress it back to a resting limit.

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import run_async

SRC = (Path(__file__).resolve().parent.parent / "main_bot.py").read_text()


# ── retry_partial_exit: a partially filled protective sell is retried ────────

def _pos(symbol, qty, market_value):
    return SimpleNamespace(symbol=symbol, qty=qty, market_value=market_value)


def _patch_exit_paths(monkeypatch, positions_seq, place_results, client=None):
    """positions_seq: successive position LISTS returned by live-position
    lookups ('ERROR' makes the lookup raise). place_results: entries for
    successive place_order calls -- True/False, or {"ok":bool, "filled":qty}
    to control the reported fill. Returns a dict recording the calls."""
    import main_bot

    calls = {"positions": 0, "orders": []}
    order_iter = iter(place_results)

    def fake_get_all_positions():
        i = calls["positions"]
        calls["positions"] += 1
        seq = positions_seq[min(i, len(positions_seq) - 1)]
        if seq == "ERROR":
            raise RuntimeError("exchange unreachable")
        return seq

    class FakeClient:
        get_all_positions = staticmethod(fake_get_all_positions)

    async def fake_place_order(symbol, side, qty, price, **kw):
        res = next(order_iter, False)
        if isinstance(res, dict):
            ok = res.get("ok", True)
            filled = res.get("filled", qty)
        else:
            ok = bool(res)
            filled = qty
        oid_out = kw.get("order_id_out")
        if oid_out is not None:
            oid_out["order_id"] = "oid-1"
            oid_out["requested_qty"] = qty
            oid_out["filled_qty"] = filled if ok else 0.0
        calls["orders"].append({"symbol": symbol, "qty": qty, **kw})
        return ok

    monkeypatch.setattr(main_bot, "trading_client", client or FakeClient())
    monkeypatch.setattr(main_bot, "place_order", fake_place_order)
    monkeypatch.setattr(main_bot, "POSITION_SETTLE_DELAY", 0.0)  # no real sleeps
    return calls


def test_retry_partial_exit_is_a_noop_when_already_flat(monkeypatch):
    calls = _patch_exit_paths(monkeypatch, positions_seq=[[]], place_results=[])
    import main_bot
    ok = run_async(main_bot.retry_partial_exit("BTC/USD", 100.0, 100.0, "Stop loss"))
    assert ok is True
    assert calls["orders"] == []


def test_retry_partial_exit_sells_the_remainder_then_confirms_flat(monkeypatch):
    # First lookup: 0.4 remaining. After the retry: flat (empty list).
    calls = _patch_exit_paths(
        monkeypatch,
        positions_seq=[[_pos("BTCUSD", 0.4, 40.0)], []],
        place_results=[True],
    )
    import main_bot
    ok = run_async(main_bot.retry_partial_exit("BTC/USD", 100.0, 100.0, "Stop loss"))
    assert ok is True
    assert len(calls["orders"]) == 1
    o = calls["orders"][0]
    assert o["qty"] == pytest.approx(0.4)   # the remainder, not the original qty
    assert o["market"] is True
    assert o["symbol"] == "BTC/USD"


def test_retry_partial_exit_matches_symbol_without_slash(monkeypatch):
    # Exchange reports slash-less 'BTCUSD'; we pass 'BTC/USD'.
    calls = _patch_exit_paths(
        monkeypatch,
        positions_seq=[[_pos("BTCUSD", 0.2, 20.0)], []],
        place_results=[True],
    )
    import main_bot
    assert run_async(main_bot.retry_partial_exit("BTC/USD", 100.0, 100.0, "Max hold")) is True
    assert len(calls["orders"]) == 1


def test_retry_partial_exit_ignores_other_symbols(monkeypatch):
    # A different symbol's position must not be mistaken for our residual.
    calls = _patch_exit_paths(
        monkeypatch,
        positions_seq=[[_pos("ETHUSD", 5.0, 500.0)], []],
        place_results=[True],
    )
    import main_bot
    assert run_async(main_bot.retry_partial_exit("BTC/USD", 100.0, 100.0, "Max hold")) is True
    assert calls["orders"] == []


def test_retry_partial_exit_leaves_unsellable_dust(monkeypatch):
    # Residual value below MIN_ORDER_USD -> cannot be sold -> treated as flat.
    import config
    calls = _patch_exit_paths(
        monkeypatch,
        positions_seq=[[_pos("BTCUSD", 1e-8, config.MIN_ORDER_USD - 1.0)]],
        place_results=[],
    )
    import main_bot
    assert run_async(main_bot.retry_partial_exit("BTC/USD", 100.0, 100.0, "Slow-bleed")) is True
    assert calls["orders"] == []


def test_retry_partial_exit_returns_false_when_retry_is_rejected(monkeypatch):
    _patch_exit_paths(
        monkeypatch,
        positions_seq=[[_pos("BTCUSD", 0.4, 40.0)]],
        place_results=[False],  # remainder sell rejected
    )
    import main_bot
    assert run_async(main_bot.retry_partial_exit("BTC/USD", 100.0, 100.0, "Stop loss")) is False


def test_retry_partial_exit_returns_false_when_lookup_fails(monkeypatch):
    # A failed position lookup must NOT be treated as flat: the position could
    # still be open, so the exit stays unconfirmed and is retried next loop.
    calls = _patch_exit_paths(monkeypatch, positions_seq=["ERROR"], place_results=[])
    import main_bot
    assert run_async(main_bot.retry_partial_exit("BTC/USD", 100.0, 100.0, "Stop loss")) is False
    assert calls["orders"] == []


def test_retry_partial_exit_bounded_attempts(monkeypatch):
    # Never flat, and the retry keeps succeeding: must stop after max_attempts.
    calls = _patch_exit_paths(
        monkeypatch,
        positions_seq=[[_pos("BTCUSD", 0.4, 40.0)]],  # always non-flat
        place_results=[True, True, True, True],
    )
    import main_bot
    ok = run_async(main_bot.retry_partial_exit("BTC/USD", 100.0, 100.0, "Stop loss", max_attempts=3))
    assert ok is False              # residual remained -> not done
    assert len(calls["orders"]) == 3  # bounded by max_attempts


# ── position-fetch lag: a reported full fill is not contradicted by an
#    immediate stale position read ──────────────────────────────────────────

def test_full_fill_with_stale_position_does_not_sell_again(monkeypatch):
    # place_order reported the whole 1.0 filled. The first position re-read is
    # stale (still shows 1.0), but the next settle read shows flat. No second
    # sell may be submitted.
    oid = {"order_id": "x", "requested_qty": 1.0, "filled_qty": 1.0}
    calls = _patch_exit_paths(
        monkeypatch,
        positions_seq=[[_pos("BTCUSD", 1.0, 100.0)], []],  # stale, then flat
        place_results=[],
    )
    import main_bot
    ok = run_async(main_bot.retry_partial_exit("BTC/USD", 100.0, 100.0, "Stop loss", order_id_out=oid))
    assert ok is True
    assert calls["orders"] == []          # no double-sell
    assert calls["positions"] >= 2        # it did wait before trusting the read


def test_full_fill_stale_for_the_whole_settle_window_then_retries(monkeypatch):
    # Reported full fill but the position genuinely still shows stock even after
    # the settle window: then (and only then) treat it as a real remainder.
    calls = _patch_exit_paths(
        monkeypatch,
        positions_seq=[
            [_pos("BTCUSD", 1.0, 100.0)],  # initial read
            [_pos("BTCUSD", 1.0, 100.0)],  # settle 1
            [_pos("BTCUSD", 1.0, 100.0)],  # settle 2
            [_pos("BTCUSD", 1.0, 100.0)],  # settle 3
            [],                            # flat after the retry
        ],
        place_results=[{"ok": True, "filled": 1.0}],
    )
    import main_bot
    oid = {"order_id": "x", "requested_qty": 1.0, "filled_qty": 1.0}
    ok = run_async(main_bot.retry_partial_exit("BTC/USD", 100.0, 100.0, "Stop loss", order_id_out=oid))
    assert ok is True
    assert len(calls["orders"]) == 1
    assert calls["orders"][0]["qty"] == pytest.approx(1.0)


def test_partial_fill_proceeds_without_waiting(monkeypatch):
    # filled 0.4 < requested 1.0 => NOT a full fill, so no settle wait: the 0.6
    # remainder is retried right away (the second buy of an already-partial fill).
    calls = _patch_exit_paths(
        monkeypatch,
        positions_seq=[[_pos("BTCUSD", 0.6, 60.0)], []],
        place_results=[{"ok": True, "filled": 0.6}],
    )
    import main_bot
    oid = {"order_id": "x", "requested_qty": 1.0, "filled_qty": 0.4}
    ok = run_async(main_bot.retry_partial_exit("BTC/USD", 100.0, 100.0, "Stop loss", order_id_out=oid))
    assert ok is True
    assert calls["positions"] == 2            # read, retry, read -> no settle polls
    assert calls["orders"][0]["qty"] == pytest.approx(0.6)


def test_unreported_fill_qty_is_not_treated_as_full(monkeypatch):
    # If the exchange never reported a fill qty (filled 0.0), we cannot claim a
    # full fill, so a non-flat position read is acted on immediately -- no
    # settle wait. max_attempts=1 isolates the first decision.
    calls = _patch_exit_paths(
        monkeypatch,
        positions_seq=[[_pos("BTCUSD", 1.0, 100.0)]],
        place_results=[{"ok": True, "filled": 1.0}],
    )
    import main_bot
    oid = {"order_id": "x", "requested_qty": 1.0, "filled_qty": 0.0}
    ok = run_async(main_bot.retry_partial_exit("BTC/USD", 100.0, 100.0, "Stop loss", order_id_out=oid, max_attempts=1))
    assert ok is False
    assert len(calls["orders"]) == 1
    # one initial read + one final read; no settle polls (which would be +3)
    assert calls["positions"] == 2


def test_exit_branch_calls_retry_partial_exit_for_protective_sells():
    # Wiring guard: the exit branch invokes retry_partial_exit after a
    # successful protective market sell, before marking the exit done.
    _find(
        r"if success and _protective:\s*\n(.*?)success = await retry_partial_exit\("
    )


def _find(pattern: str) -> re.Match:
    m = re.search(pattern, SRC, re.DOTALL)
    assert m is not None, f"pattern not found in main_bot.py: {pattern!r}"
    return m


def test_regular_exit_branch_submits_a_market_sell_of_qty_held():
    # The stop-loss / slow-bleed / max-hold / weak-signal exit branch routes
    # through is_protective_exit() and sells qty_held.
    _find(
        r"place_order\(\s*symbol,\s*OrderSide\.SELL,\s*qty_held,.*?market=_protective\s*\)"
    )


def test_protective_exit_classification():
    from main_bot import is_protective_exit
    # Risk exits -> market.
    assert is_protective_exit("🛑 Time-Decay Stop loss (-2.10% <= -1.00%) [normal]")
    assert is_protective_exit("🐌 Slow-bleed exit (-1.20% after 1.5h, never reached +2% ...)")
    assert is_protective_exit("⏰ Max hold time (4.1h)")
    assert is_protective_exit("📉 Trailing Stop triggered (Peak: $110.00, Stop: 1.00% off peak ...)")
    # Discretionary -> resting limit.
    assert not is_protective_exit("📉 Signal weak (0.420) [normal]")
    assert not is_protective_exit("")
    assert not is_protective_exit(None)


def test_kill_switch_submits_a_market_sell_of_the_position_qty():
    # Daily-loss kill switch: emergency flatten uses p["qty"] from the
    # position, not a qty remembered from the buy.
    _find(
        r"place_order\(\s*denormalize_symbol\(symbol\),\s*OrderSide\.SELL,\s*"
        r"float\(p\[\"qty\"\]\),.*?market=True\s*\)"
    )


def test_entries_are_not_marked_market():
    # The BUY submission must not pass market=True -- entries stay limits.
    m = _find(r"place_order\(symbol,\s*OrderSide\.BUY,[^)]*\)")
    assert "market=True" not in m.group(0)


def test_failed_exit_sell_logs_loudly_and_does_not_mark_pending():
    # A failed protective sell must (a) log an error and (b) NOT mark the
    # position pending, so the same exit condition re-fires and is retried.
    m = _find(
        r"if not success:\s*\n(.*?)\n\s*if success:"
    )
    body = m.group(1)
    assert "logger.error" in body
    assert "mark_pending_exit" not in body
