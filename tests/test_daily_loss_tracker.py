# tests/test_daily_loss_tracker.py — the DAILY_LOSS_LIMIT kill-switch must
# measure loss against the current UTC day's opening equity, not the equity at
# process start. Regression guard for the bug where start_equity was pinned
# once and "daily" losses accumulated across the whole run.

import time

from conftest import run_async  # noqa: F401  (ensures repo import shims load)
import main_bot


def _epoch(y, m, d, hh=0, mm=0):
    return time.mktime((y, m, d, hh, mm, 0, 0, 0, 0)) - time.timezone


def test_baseline_set_once_per_day():
    t = main_bot.DailyLossTracker(now=_epoch(2026, 10, 9, 0, 5))
    assert t.equity(now=_epoch(2026, 10, 9, 0, 5)) is None
    assert t.update(100_000.0, now=_epoch(2026, 10, 9, 0, 6)) == 100_000.0
    # a later reading the same day must NOT move the baseline
    assert t.update(98_000.0, now=_epoch(2026, 10, 9, 6, 0)) == 100_000.0


def test_nonpositive_equity_never_becomes_baseline():
    t = main_bot.DailyLossTracker(now=_epoch(2026, 10, 9))
    assert t.update(0.0, now=_epoch(2026, 10, 9, 1)) is None
    assert t.update(50_000.0, now=_epoch(2026, 10, 9, 2)) == 50_000.0


def test_new_utc_day_resets_baseline():
    t = main_bot.DailyLossTracker(now=_epoch(2026, 10, 9, 23, 50))
    t.update(100_000.0, now=_epoch(2026, 10, 9, 23, 50))
    assert t.roll(now=_epoch(2026, 10, 10, 0, 5)) is True
    assert t.equity(now=_epoch(2026, 10, 10, 0, 5)) is None
    assert t.update(95_000.0, now=_epoch(2026, 10, 10, 0, 6)) == 95_000.0


def test_yesterdays_loss_does_not_trip_todays_limit():
    """The core regression: down 5% from process start is harmless if today
    opened flat. Under the old code this -5% persisted into the next day."""
    day1 = _epoch(2026, 10, 9, 12)
    t = main_bot.DailyLossTracker(now=day1)
    t.update(100_000.0, now=day1)                       # day 1 opens at 100k
    day1_loss = (95_000.0 - t.equity(now=day1)) / t.equity(now=day1) * 100
    assert day1_loss <= -4.9                            # down ~5% on day 1

    day2 = _epoch(2026, 10, 10, 12)
    t.update(95_000.0, now=day2)                        # day 2 opens at 95k
    day2_loss = (95_000.0 - t.equity(now=day2)) / t.equity(now=day2) * 100
    assert day2_loss == 0.0                             # no phantom carry-over


def test_same_day_limit_still_fires():
    """The fix must not weaken the same-day protection."""
    t = main_bot.DailyLossTracker(now=_epoch(2026, 10, 9, 0, 5))
    t.update(100_000.0, now=_epoch(2026, 10, 9, 0, 5))
    day_start = t.equity(now=_epoch(2026, 10, 9, 12))
    loss = (96_500.0 - day_start) / day_start * 100
    assert loss <= -3.0                                 # <= DAILY_LOSS_LIMIT
