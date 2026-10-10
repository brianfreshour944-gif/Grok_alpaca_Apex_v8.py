# Donchian paper-forward ledger — the honest live test

This is the forward test for the **one strategy Alpaca can actually run**
(long-only spot crypto). It costs nothing and puts nothing at risk: the cron
appends a hypothetical daily book and the harness marks it to market.

## What was missing

`donchian_bot.py --paper --ledger` already wrote a daily row of the *target*
book — but never tracked whether that book made money. A ledger of intentions
is not a performance record. `research/donchian_forward.py` closes that gap:
it holds each day's book until the next row, charges turnover costs, and reports
forward performance against **equal-weight buy & hold of the same names**.

## The bug this caught

First run showed the book at **+174%, Sharpe 3.82** — absurdly strong for a
period the yearly backtest called flat. Cause: venue bars are stamped
**16:00 UTC** while ledger dates are midnight, so the mark compared mismatched
days and could run *backwards*. After flooring both to calendar dates:
**−5.9%**, not +174%. A forward harness is only as honest as its date alignment.

## Result (backfilled 400 days to 2026-10-09, 25 bps)

| | total | Sharpe | maxDD |
|---|---|---|---|
| **donchian book (enhanced default)** | **−5.9%** | −0.12 | −29.1% |
| donchian book (base rule) | −28.2% | −0.75 | −44.6% |
| equal-weight buy & hold | −44.2% | −0.73 | −66.5% |

Read it honestly: over this window everything lost money — it was a down tape.
The strategy **lost a little and lost far less than holding** (−6% vs −44%), and
the promoted *enhanced* default clearly beat the base rule (−6% vs −28%). That
is exactly what a long-only trend rule should do in a falling market, and it is
consistent with the yearly backtest.

## How to run

```bash
# one-shot: rebuild the ledger the cron would have accrued, then mark it
python research/donchian_forward.py --ledger donchian_ledger.csv \
    --cache okx_daily --backfill-days 400          # --base for the textbook rule

# or just report an existing (cron-accrued) ledger
python research/donchian_forward.py --ledger donchian_ledger.csv --cache okx_daily
```

`research/run_donchian_daily.sh` now fetches OKX bars, appends the paper row, and
prints the forward report — so each day's cron run shows whether the book is
winning, losing, or beating simply holding.

## What to watch for

- **Beating buy & hold in down tapes** is the expected, honest outcome for a
  long-only trend rule. That is not proof of alpha — it is proof the stop works.
- **The real question is the up tape.** If the book underperforms buy & hold
  when crypto rises (as the backtest suggests it will), then holding is the
  better strategy for a small account and the bot is a learning tool, not a
  money-maker.
- **20+ days minimum** before reading anything into it; the harness prints a
  warning below that.
- Alpaca crypto is spot-only, so the book is long-only by construction — there
  is no short leg to worry about.

This is deliberately the last piece of research. It puts the only placeable
strategy on live data, at zero risk, and lets the tape — not another backtest —
decide.
