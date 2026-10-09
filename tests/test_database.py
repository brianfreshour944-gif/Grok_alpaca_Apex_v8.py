# tests/test_database.py — database.py, with psycopg2.connect mocked out
# (no real Postgres needed). This exists specifically to check the thing the
# repo's old audit scripts never did: that the SQL placeholder count matches
# the values tuple, and that the right values land in the right columns --
# not just that some INSERT statement is present in the source text.

import os
from unittest.mock import MagicMock

import pytest

import database


@pytest.fixture
def mock_db(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake/db")
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.__enter__.return_value = mock_cursor
    monkeypatch.setattr(database.psycopg2, "connect", MagicMock(return_value=mock_conn))
    return mock_cursor


def _last_insert_trades_call(mock_cursor):
    for call in mock_cursor.execute.call_args_list:
        sql = call.args[0]
        if "INSERT INTO trades" in sql:
            return call.args[0], call.args[1]
    raise AssertionError("no INSERT INTO trades call found")


def test_record_trade_placeholder_count_matches_values_tuple(mock_db):
    database.record_trade(
        "bot", "BTC/USD", "sell", 1.0, 100.0,
        order_id="oid-1", fee=0.5, fill_price=101.0,
        realized_pnl=19.5, realized_pnl_pct=0.1,
    )
    sql, values = _last_insert_trades_call(mock_db)
    # one literal 'Alpaca' for exchange, one NOW() for timestamp -- everything
    # else in the column list must be a %s with a matching value.
    assert sql.count("%s") == len(values)


def test_record_trade_stores_realized_pnl_in_the_right_position(mock_db):
    database.record_trade(
        "bot", "BTC/USD", "sell", 1.0, 100.0,
        order_id="oid-2", fee=0.5, fill_price=101.0,
        realized_pnl=19.5, realized_pnl_pct=0.1,
    )
    sql, values = _last_insert_trades_call(mock_db)
    columns = sql.split("(", 1)[1].split(")", 1)[0]
    columns = [c.strip() for c in columns.split(",")]
    # columns list includes the literal 'Alpaca' position (exchange) and
    # NOW() (timestamp), which don't have a corresponding %s -- build the
    # %s-only column list to zip against `values`.
    literal_cols = {"exchange", "timestamp"}
    placeholder_cols = [c for c in columns if c not in literal_cols]
    row = dict(zip(placeholder_cols, values))
    assert row["realized_pnl"] == pytest.approx(19.5)
    assert row["realized_pnl_pct"] == pytest.approx(0.1)


def test_record_trade_buy_has_null_realized_pnl(mock_db):
    database.record_trade("bot", "BTC/USD", "buy", 1.0, 100.0, order_id="oid-3", fee=0.1, fill_price=100.1)
    sql, values = _last_insert_trades_call(mock_db)
    columns = [c.strip() for c in sql.split("(", 1)[1].split(")", 1)[0].split(",")]
    placeholder_cols = [c for c in columns if c not in {"exchange", "timestamp"}]
    row = dict(zip(placeholder_cols, values))
    assert row["realized_pnl"] is None
    assert row["realized_pnl_pct"] is None


def test_record_trade_stores_the_filled_qty_on_a_partial_fill(mock_db):
    """The stored quantity (and the value derived from it) must be the qty that
    actually filled, not the qty requested -- otherwise a partial fill inflates
    the position and the value column."""
    database.record_trade(
        "bot", "BTC/USD", "sell", 1.0, 109.9, order_id="oid-part",
        fee=0.21, fill_price=110.0, filled_qty=0.4,
    )
    sql, values = _last_insert_trades_call(mock_db)
    columns = [c.strip() for c in sql.split("(", 1)[1].split(")", 1)[0].split(",")]
    placeholder_cols = [c for c in columns if c not in {"exchange", "timestamp"}]
    row = dict(zip(placeholder_cols, values))
    assert row["quantity"] == pytest.approx(0.4)
    assert row["value"] == pytest.approx(110.0 * 0.4)


def test_record_trade_falls_back_to_requested_qty_when_no_fill_qty(mock_db):
    database.record_trade(
        "bot", "BTC/USD", "buy", 1.5, 100.0, order_id="oid-no-fill",
        fee=0.0, fill_price=100.0,
    )
    sql, values = _last_insert_trades_call(mock_db)
    columns = [c.strip() for c in sql.split("(", 1)[1].split(")", 1)[0].split(",")]
    placeholder_cols = [c for c in columns if c not in {"exchange", "timestamp"}]
    row = dict(zip(placeholder_cols, values))
    assert row["quantity"] == pytest.approx(1.5)


def test_record_trade_is_a_noop_without_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    mock_connect = MagicMock()
    monkeypatch.setattr(database.psycopg2, "connect", mock_connect)
    database.record_trade("bot", "BTC/USD", "sell", 1.0, 100.0)
    assert not mock_connect.called


def test_record_trade_persists_commission_estimated_flag(mock_db):
    database.record_trade(
        "bot", "BTC/USD", "sell", 1.0, 100.0,
        order_id="oid-est", fee=0.5, fill_price=101.0,
        realized_pnl=19.5, realized_pnl_pct=0.1, commission_estimated=True,
    )
    sql, values = _last_insert_trades_call(mock_db)
    columns = [c.strip() for c in sql.split("(", 1)[1].split(")", 1)[0].split(",")]
    placeholder_cols = [c for c in columns if c not in {"exchange", "timestamp"}]
    row = dict(zip(placeholder_cols, values))
    assert row["commission_estimated"] is True
    assert row["fee"] == pytest.approx(0.5)


def test_record_trade_defaults_commission_estimated_to_false(mock_db):
    database.record_trade("bot", "BTC/USD", "buy", 1.0, 100.0, order_id="oid-plain")
    sql, values = _last_insert_trades_call(mock_db)
    columns = [c.strip() for c in sql.split("(", 1)[1].split(")", 1)[0].split(",")]
    placeholder_cols = [c for c in columns if c not in {"exchange", "timestamp"}]
    row = dict(zip(placeholder_cols, values))
    assert row["commission_estimated"] is False


# ── backfill_trade_if_missing: estimated fee on crash-recovery rows ──

def _backfill_insert_row(mock_db):
    sql, values = _last_insert_trades_call(mock_db)
    columns = [c.strip() for c in sql.split("(", 1)[1].split(")", 1)[0].split(",")]
    placeholder_cols = [c for c in columns if c not in {"exchange", "timestamp"}]
    return dict(zip(placeholder_cols, values))


def test_backfill_estimates_fee_when_exchange_reports_none(mock_db):
    from types import SimpleNamespace
    # No commission attribute -> exchange-reported fee is absent.
    order = SimpleNamespace(
        id="bf-1", filled_qty="2.0", filled_avg_price="110.0",
        side="sell", symbol="BTCUSD", created_at=None,
    )
    mock_db.fetchone.return_value = None  # not already recorded
    assert database.backfill_trade_if_missing(order) is True

    row = _backfill_insert_row(mock_db)
    # 110 * 2 * 25 bps = 0.55 estimated
    assert row["fee"] == pytest.approx(0.55)
    assert row["commission_estimated"] is True


def test_backfill_keeps_exchange_reported_commission_when_present(mock_db):
    from types import SimpleNamespace
    order = SimpleNamespace(
        id="bf-2", filled_qty="2.0", filled_avg_price="110.0", commission="1.23",
        side="sell", symbol="BTCUSD", created_at=None,
    )
    mock_db.fetchone.return_value = None
    assert database.backfill_trade_if_missing(order) is True

    row = _backfill_insert_row(mock_db)
    assert row["fee"] == pytest.approx(1.23)
    assert row["commission_estimated"] is False


def test_backfill_skips_when_row_already_exists(mock_db):
    from types import SimpleNamespace
    order = SimpleNamespace(
        id="bf-3", filled_qty="1.0", filled_avg_price="100.0",
        side="buy", symbol="BTCUSD", created_at=None,
    )
    mock_db.fetchone.return_value = (1,)  # already recorded
    assert database.backfill_trade_if_missing(order) is False


# ── commission_estimated migration against a REAL Postgres ──
# Opt-in: set TEST_DATABASE_URL to a throwaway database (e.g.
# "postgresql://apex:apex@localhost:5432/apex_test"). Skipped by default so the
# suite needs no Postgres, but runnable to prove the migration is additive and
# idempotent against a pre-existing trades table that already holds rows.

@pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"),
    reason="set TEST_DATABASE_URL to run the real-Postgres migration test",
)
def test_commission_estimated_migration_is_additive_and_idempotent(monkeypatch):
    import psycopg2
    db = os.environ["TEST_DATABASE_URL"]
    monkeypatch.setenv("DATABASE_URL", db)

    with psycopg2.connect(db) as conn, conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS trades")
        cur.execute("""CREATE TABLE trades (
            id SERIAL PRIMARY KEY, bot_name TEXT, exchange TEXT, symbol TEXT,
            side TEXT, price REAL, quantity REAL, value REAL, fee REAL,
            fill_price REAL, order_id TEXT, timestamp TIMESTAMP)""")
        cur.execute("""INSERT INTO trades
            (bot_name,exchange,symbol,side,price,quantity,value,fee,fill_price,order_id,timestamp)
            VALUES ('b','Alpaca','BTC/USD','sell',100,1,100,0.0,101,'old-1',NOW()),
                   ('b','Alpaca','ETH/USD','buy',50,2,100,0.0,50,'old-2',NOW())""")
        conn.commit()
        cur.execute("""SELECT column_name FROM information_schema.columns
                       WHERE table_name='trades'""")
        before = {r[0] for r in cur.fetchall()}
        assert "commission_estimated" not in before

    database.init_db()
    database.init_db()  # idempotent: a second run must not error or duplicate

    with psycopg2.connect(db) as conn, conn.cursor() as cur:
        cur.execute("""SELECT column_name FROM information_schema.columns
                       WHERE table_name='trades'""")
        after = {r[0] for r in cur.fetchall()}
        cur.execute("SELECT order_id, fee, commission_estimated FROM trades ORDER BY id")
        rows = cur.fetchall()

    assert "commission_estimated" in after
    assert before.issubset(after)          # additive: nothing dropped
    assert len(rows) == 2                  # rows preserved
    assert all(r[2] is None for r in rows) # existing rows untouched (NULL)
    assert rows[0][1] == 0.0               # existing fee preserved


# ── save_bot_state: dynamic universe means the symbol set isn't fixed ──

def test_save_bot_state_persists_a_symbol_outside_any_fixed_list(mock_db):
    """
    With a dynamic universe, save_bot_state can't just iterate config.SYMBOLS
    (a static 3-symbol list) -- it must persist state for whatever symbols
    actually have state, including ones a static list would never contain.
    """
    database.save_bot_state(
        cooldown_until={"XRP/USD": 100.0},
        entry_time={"XRP/USD": 90.0},
        latest_signals={},
        highest_prices={},
    )
    calls = [c for c in mock_db.execute.call_args_list if "INSERT INTO bot_state" in c.args[0]]
    assert len(calls) == 1
    assert calls[0].args[1][0] == "XRP/USD"


def test_save_bot_state_persists_the_union_of_all_four_dicts(mock_db):
    database.save_bot_state(
        cooldown_until={"BTC/USD": 1.0},
        entry_time={"ETH/USD": 2.0},
        latest_signals={"SOL/USD": 0.6},
        highest_prices={"DOGE/USD": 0.1},
    )
    symbols_written = {
        c.args[1][0] for c in mock_db.execute.call_args_list if "INSERT INTO bot_state" in c.args[0]
    }
    assert symbols_written == {"BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"}


def test_get_realized_pnl_summary_returns_none_without_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert database.get_realized_pnl_summary("bot") is None


def test_get_realized_pnl_summary_returns_summed_values(mock_db):
    mock_db.fetchone.return_value = (123.45, 6.78, 9)
    result = database.get_realized_pnl_summary("bot")
    assert result == {"total_realized_pnl": 123.45, "total_fees": 6.78, "closed_trades": 9}


def test_get_realized_pnl_summary_filters_by_bot_name_and_sell_side(mock_db):
    mock_db.fetchone.return_value = (0.0, 0.0, 0)
    database.get_realized_pnl_summary("my-bot")
    sql, params = mock_db.execute.call_args.args
    assert "side = 'sell'" in sql
    assert params == ("my-bot",)


def test_report_equity_is_a_noop_without_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert database.report_equity("bot", 1000.0) is False
