# Backtest exploration: chasing a "winning percentage"

Branch: `backtest/winning-pct-experiments` (off `main` @ b57149c). **No config
was changed; nothing pushed.** This documents a real-data parameter search on
the in-repo live-parity backtester (`backtest_apex.py`).

## Setup

- Real 15-minute Alpaca crypto bars (same venue the live bot uses; needs no key),
  2025-04-01 -> 2026-10-08 (~555 days), 10 trained symbols (5 active at a time).
- Fees 25 bps / slippage 5 bps per fill (Alpaca tier-1 taker) unless noted.
- Out-of-sample cutoff = the model file's last git commit (2026-07-19); anything
  after it is genuinely unseen by the model.
- Timing-alpha null control: `--null-trials`, which circularly shifts each
  symbol's signal by a random >=3-day offset (same distribution/autocorrelation,
  no link to price) and re-runs the whole simulator.
- Harness: `sweep_harness.py` loads bars + stage-1 signals once, then varies
  `config` per run (the simulator reads config at cycle time), so a sweep is
  seconds, not minutes.

## Baseline (current `main` config)

| metric | value |
|---|---|
| Total return | **-3.01%** |
| Win rate | 24.4% |
| Mean net / trade | -56.2 bps |
| Mean gross / trade | -6.3 bps |
| Profit factor | 0.28 |
| Trades | 480 (live kill-switch halted at 2025-04-18; ~14k if unconstrained) |

Two structural facts dominate:

1. **Fees are ~25 bps per leg; the raw signal's per-trade edge is single-digit
   bps.** Frictionless the whole book is +3.24%, PF 1.07, 54.4% win -- barely
   positive. Costs turn it decisively negative.
2. **The live kill-switch is a permanent -3% stop-out** (`start_equity` is set
   once), so the default run quits after ~2 weeks. Unconstrained, the strategy
   loses ~23%.

## What was swept

- `BUY_SIGNAL` (entry selectivity) 0.51 -> 0.90.
- Exits: `MAX_HOLD_HOURS` 4h -> 48h, `STOP_LOSS_PCT` 0.02 -> 0.06,
  `SLOW_BLEED_PCT` on/off, trailing min/max, `MIN_HOLD_HOURS_BEFORE_SIGNAL`.
- `BUY_SIGNAL x MAX_HOLD` grid.
- Kill-switch mode (off / daily / live).

## Results (kill-switch off, unconstrained; 10 symbols)

| BUY_SIGNAL | n | win% | ret% | net/trade | PF | OOS n | OOS ret% |
|---|---|---|---|---|---|---|---|
| 0.51 | 15126 | 18.2 | -24.15 | -57.5 | 0.22 | 2342 | -3.44 |
| 0.55 | 7557 | 23.8 | -15.13 | -55.1 | 0.31 | 1107 | -1.69 |
| 0.60 | 1575 | 28.1 | -8.28 | -57.4 | 0.38 | 185 | -0.43 |
| 0.62 | 676 | 33.0 | -7.29 | -52.0 | 0.42 | 60 | -0.53 |
| 0.66 | 180 | 37.8 | -3.18 | -39.1 | 0.60 | **7** | +1.12 |
| 0.70 | 84 | 48.0 | -0.03 | -1.0 | 0.99 | **2** | +0.56 |

Exit tuning moves the book only from -24% to -21%; **gross per-trade stays
~-7 bps in every exit configuration**, and every OOS bucket with a real sample
stays ~-3%.

## The "winning" numbers are not real edge

1. **`BUY_SIGNAL > 0.70` is the only setting near break-even** (48-51% win,
   PF ~1.0) -- but it fires **only 2 times out-of-sample** across the entire
   15-month unseen window. The headline is ~99% in-sample.
2. **The model is stale.** Its >0.70 signals cluster before the 2026-07-19
   cutoff and almost vanish after it. A threshold selected on that period is
   not being validated; it is being fit to the training era.
3. **Null test says no timing alpha.** At `BUY_SIGNAL > 0.60` (the best OOS
   cell with a usable sample) the circular-shift null gives one-sided
   p = 0.46 -- random signal re-runs do as well. At the 0.51 baseline p = 0.08;
   only the n<=7 >0.66 cell is "significant", which is meaningless at that size.
4. **Win rate is a trap.** The gates can push win% from 18% toward 35-50%, but
   fees keep PnL negative. Win% rises by taking smaller, rarer bets, not by
   finding an edge.

## Verdict

**A "winning percentage" is achievable in-sample by tuning `BUY_SIGNAL` (and,
weakly, hold time), but it does not survive out-of-sample, does not clear the
timing-alpha null control, and does not clear costs.** With the current model
and a 25 bps taker fee, the lever that would make this profitable is the model's
edge per trade, not any config value in this repo. The single genuinely useful
change is **operational**: the default `live` kill-switch is a permanent -3%
stop-out (`start_equity` set once), which is why the shipped configuration just
goes quiet -- worth fixing independently of any edge question.

## Independence / limits

- One data realisation; no IS/OOS sharding across time beyond the model cutoff.
- Orderbook whale filter is not modelled (can only remove trades).
- Survivorship-biased 10-symbol pool, as `config.py` itself flags.
- Live kill-switch behaviour is modelled (`start_equity` never resets).
