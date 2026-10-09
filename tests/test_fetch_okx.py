# tests/test_fetch_okx.py — OKX fetcher parsing/schema (no network).
import csv

import research.fetch_okx as ok


def test_inst_mapping():
    assert ok._insts("BTCUSDT") == ("BTC-USDT-SWAP", "BTC-USDT")
    assert ok._insts("ATOMUSDT") == ("ATOM-USDT-SWAP", "ATOM-USDT")


def test_fetch_funding_schema(tmp_path, monkeypatch):
    pages = [
        {"code": "0", "data": [
            {"fundingTime": "1791561600000", "realizedRate": "0.0001"},
            {"fundingTime": "1791532800000", "realizedRate": "0.0002"},
        ]},
        {"code": "0", "data": []},
    ]
    monkeypatch.setattr(ok, "_get", lambda url, tries=4: pages.pop(0))
    n = ok.fetch_funding("BTCUSDT", tmp_path, days=3650)
    assert n == 2
    rows = list(csv.reader((tmp_path / "funding" / "BTCUSDT.csv").open()))
    assert rows[0] == ["ts", "funding_rate"]
    assert rows[1][0].isdigit() and rows[1][1] == "0.0002"   # sorted ascending


def test_fetch_candles_writes_ms_epoch(tmp_path, monkeypatch):
    pages = [
        {"code": "0", "data": [
            ["1791561600000", "100", "110", "90", "105", "5"],
            ["1791558000000", "99", "101", "98", "100", "3"],
        ]},
        {"code": "0", "data": []},
    ]
    monkeypatch.setattr(ok, "_get", lambda url, tries=4: pages.pop(0))
    n = ok.fetch_candles("BTCUSDT", False, tmp_path, days=3650)
    assert n == 2
    rows = list(csv.reader((tmp_path / "BTCUSDT_1h.csv").open()))
    assert rows[0] == ok.KLINE_COLS
    assert rows[1][0] == "1791558000000"                     # ascending, ms epoch
    assert len(rows[1]) == 6
