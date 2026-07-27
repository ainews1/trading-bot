# Code limitations sweep — correctness + perf fixes (2026-06-10)

Investigation: 3 parallel analysis agents (orchestration / strategy+SMC / data layer+logs)
+ manual verification against code + live API probe. Bot RUNNING (paper, +$40 today);
edits take effect only on restart.

## Scope decision
Only correctness + computational fixes that align live behavior with the validated backtest.
Signal-quality changes (volume filter on all entry types, ATR-based EMA tolerance, HTF
confluence, SMC threshold tuning, 4h migration) change trading behavior → deferred until
backtested (prior verdict: 5m has ~0 gross edge; edge needs entry-quality work, not tweaks).

## Tasks
- [x] 1. HIGH fix: drop still-forming partial candle in fetch_ohlcv (VERIFIED via live probe:
      Poloniex returns the current partial 5m interval as last row → analyze() runs on a
      2-second-old bar and check_paper_exit never sees the closed candle's wicks)
- [x] 2. HIGH fix: daily-loss-limit denominator — use day-opening balance (persisted),
      not current balance (bot.py:637)
- [x] 3. Guard: empty OHLCV response → return None (avoids IndexError crash path)
- [x] 4. PERF: vectorize sqz momentum linreg (per-bar np.polyfit loop → precomputed
      weight vector dot product) — must be numerically identical
- [x] 5. PERF: True Range via np.maximum instead of pd.concat().max(axis=1)
- [x] 6. Equivalence test (old vs new momentum/TR on real data) + run test_regime_gate.py
- [x] 7. Review section; note restart required (restart only when no open position)

## Review (2026-06-10)
Files changed: bot.py, strategy_sqzmom_smc.py. New tests: test_squeeze_equivalence.py,
test_bot_fixes.py. Live bot process NOT touched; changes apply on next restart.

1. **Partial-candle bug (the big one).** fetch_ohlcv now drops the forming candle
   (`_timeframe_seconds()` helper, shared with wait_for_next_candle). Before: every
   loop analyzed a ~2-second-old partial bar (near-zero volume → SQUEEZE_FIRE entries
   nearly impossible live; MOM_ACCEL/TREND_CONT fired off boundary-tick prices), and
   SL/TP wick checks looked at the partial bar — the just-closed candle's high/low was
   NEVER examined. Live behavior now matches the validated backtest semantics.
   ⚠ Expect live trade pattern to shift after restart: more SQUEEZE_FIRE entries,
   exits that respect intra-candle wicks. This is the bug fix, not a regression.
2. **Daily loss cap.** `_daily_loss_pct()` divides by `daily_open_balance` (persisted
   in paper_state.json; reconstructed as balance−daily_pnl for legacy state files).
   Previously the cap loosened as balance fell during the day.
3. **Perf.** Momentum linreg vectorized (weights precomputed in __init__):
   22.5ms → 2.3ms per 250-candle analyze; 120x on 5k-candle frames (backtests).
   Equivalence proven to 1e-9 on real historical data incl. live config params,
   NaN-warmup frames, and frames shorter than mom_length. TR pd.concat → np.maximum
   (with fillna(high−low) on first bar to keep skipna semantics identical).
4. Verification: test_squeeze_equivalence.py 6/6, test_bot_fixes.py 11/11,
   test_regime_gate.py 8/8 + fail-open, py_compile clean. Tests run in temp dir —
   live paper_state.json/log untouched.

Rejected agent findings (verified false on code read): save-ordering, stale balance,
stale regime snapshot. Pyright noise in bot.py is pre-existing (Optional exchange,
hasattr-guarded calls, fcntl on win32).

## Deferred (needs backtest before shipping — listed for user)
- Volume confirmation on MOM_ACCEL / TREND_CONT entries
- EMA tolerance as ATR-based instead of fixed 0.1%
- HTF (4h) trend confluence gate; 4h timeframe migration (break-even at taker fees)
- SMC bias threshold tuning / OB age filter / swing-uniqueness relaxation
- NOTE: the partial-candle fix invalidates prior PAPER results as a baseline —
  paper PnL before/after restart are not comparable.

## Verified-NOT-issues (agent findings rejected on code read)
- Paper position save ordering: state already saved before alert (bot.py:393-394)
- Stale balance into analyze(): balance fetched at line 660, used at 664 — fresh
- Regime snapshot staleness: scout interval == loop cadence (5m) — refreshed every loop

---

# Regime Filter — block counter-trend & ranging entries

## Problem (data-confirmed)
Bot is NOT worse in downtrends. Analysis of 882 SQZMOM_SMC paper trades (`analyze_trend_perf.py`):
- By BTC price trend: UP +$3.08/trade, DOWN +$3.29/trade, FLAT/ranging **−$1.85/trade**.
- Real driver = trend *alignment*: with-trend trades win; counter-trend longs (−$542) and
  counter-trend shorts (−$300) lose. Ranging markets are the worst bucket.
- Counterfactual (block counter-trend + RANGING): removes 217 trades worth −$2,376,
  expectancy +$1.65 → **+$5.76/trade** (last-35d: +$0.06 → +$1.76/trade).

## Design
Scout already computes `trend_direction` / `market_regime` but only logs them.
Gate at the orchestration layer (`bot.py`), keeping the strategy a pure signal generator.
Config-toggleable for rollback + A/B.

## Tasks
- [x] config.py: add REGIME_FILTER_ENABLED, BLOCK_COUNTER_TREND, BLOCK_RANGING_REGIME,
      COUNTER_TREND_MIN_STRENGTH
- [x] bot.py: store `self.last_snapshot` whenever scout runs
- [x] bot.py: add `_regime_gate(signal)` -> Optional[str] (block reason)
- [x] bot.py: apply gate after `analyze()`, log `[REGIME BLOCK]`, drop signal (fail-open if no snapshot)
- [x] Verify: unit check of gate logic (test_regime_gate.py — 8/8 + fail-open + threshold) + py_compile
- [x] Counterfactual projected-impact captured in analyze_trend_perf.py

## Notes
- LONG opposed by trend_direction=="BEAR"; SHORT opposed by "BULL".
- REVERSAL regime left tradeable (no evidence it's a loser; may hold with-trend winners).
- This is in-sample estimate; a full strategy backtest with the filter is the next validation step.

## Review
Implemented as designed. Files changed:
- config.py: 4 new toggles (filter on by default).
- bot.py: `self.last_snapshot` stored on each scout; `_regime_gate()` helper; gate applied
  after `analyze()` with `[REGIME BLOCK]` logging. Strategy untouched (pure signal generator).
- test_regime_gate.py: 8 behavioral cases + fail-open (disabled flag / no snapshot) + strength
  threshold — all pass. bot.py & config.py byte-compile clean.

Projected impact (in-sample counterfactual, 882 trades): expectancy +$1.65 -> +$5.76/trade;
last-35d +$0.06 -> +$1.76/trade.

Rollback: set REGIME_FILTER_ENABLED = False.

## Out-of-sample backtest (DONE) — backtest_sqzmom_gate.py + validate_backtest.py
Data 2022-01..2026-02-05 (431k 5m candles), ends BEFORE the paper period -> genuine OOS.
Reimplementation validated vs real analyze(): 2066/2066 signals matched, 0 conflicts (PASS).
Fixed $20 risk/trade to isolate per-trade edge (2% compounding death-spiraled to $0 — artifact).

Results:
- GROSS: gate-off -0.000R, gate-on -0.008R  -> strategy has ~ZERO edge before costs.
- NET (0.14% round-trip): gate-off -1.30R, gate-on -1.25R -> catastrophic.
- Gate helps consistently (every year less-bad, ~20% smaller net loss) but cannot create edge.
- Fee sensitivity: even 0.02% round-trip = -0.19R; realistic taker 0.05-0.07% = -0.46..-0.64R.
  Break-even needs ~0% fees -> gross edge too thin to survive ANY real cost.
- Root cause: ~40 trades/day with tight 1.2x ATR(10) stops -> notional ~500-900x risk$ ->
  percentage fees become ~1R/trade. Overtrading + tight stops, not direction.

VERDICT: keep the gate ON (strict improvement), but DO NOT go live. The strategy needs
fundamental rework (far fewer/higher-quality trades, wider stops/targets vs fees, maker/limit
orders not market, higher timeframe). Paper +$1455 gross was a small favorable-window sample
(partial uptime, daily-loss-limit pauses) — not a durable edge.

Backtest caveats: omits live daily-loss-limit pause + downtime (inflates trade count/$ loss,
not per-trade expectancy); omits SMC gate (validated negligible, 1/2067); no funding fees
(would worsen); SL-priority-on-tie (pessimistic); fixed-risk by design.

## Structural-rework experiment (DONE) — backtest_variants.py
Swept timeframe x stop-width x fee x gate (same validated engine). Total R over 2022-2026:

  TF  stops    gate    n    win%  grossR  makerR(0.02%)  takerR(0.10%)  notional/risk
  5m  1.2/3.6  ON    50269  31.5   -393     -9479         -45051         903x   <- catastrophic
  1h  2.5/5.0  ON     1649  43.2   +88      +63           -38            77x
  4h  1.2/3.6  ON     1001  34.5   +90      +73           +5             85x   <- best @ taker
  4h  1.2/3.6  off    1068  33.7   +72      +54           -20            86x

Findings:
- Higher timeframe is the real fix: notional/risk collapses 903x (5m) -> ~80x (1h/4h),
  so percentage fees stop dwarfing the risk. Trades drop 60k -> ~1000.
- Gross edge is small but POSITIVE on 1h/4h (5m was ~0) -> a thin edge exists; 5m fees hid it.
- Gate helps in nearly every variant; flips 4h/taker from -20R to +5R.
- REALISTIC (taker 0.10%, market orders) best case = 4h + gate ON = +5R total over 3.5yr
  = ~+0.005R/trade. Break-even, not a money-maker. Maker (0.02%) is clearly positive but
  assumes limit entries fill — dubious for breakout entries (treat as ceiling, taker as floor).

VERDICT: rework direction validated (4h + gate removes the fee catastrophe), but the edge is
thin/break-even at realistic fees. The strategy needs a stronger ENTRY edge, not just lower
frequency. Do NOT go live. Next: better entry filter to lift expectancy, walk-forward on stop
mults (avoid overfit), add funding cost, model maker-fill realism.

# Session 2026-06-11 — "bot performs much worse" investigation + profitability rework

## Investigation (DONE)
- [x] Verify 06-10 fixes bug-free: test_bot_fixes 11/11, squeeze equivalence PASS, gate 8/8
- [x] Quantify pre/post restart: Jun1-9 avg -22/day (already losing), Jun10-11 avg -62/day
- [x] Root cause: NOT a new bug. Partial-candle fix unmasked true strategy expectancy
      (26% WR @ 2.1:1 payoff = -0.19R/trade incl fees, ~20 trades/day = -$60/day)
- Conclusion matches 06-03 backtest verdict: 5m config cannot survive fees.

## Profitability rework (plan)
- [ ] 1. Build backtest_entry_filters.py: validated variants engine + walk-forward
      (train 2022-2024, validate 2025-2026/02), 4h focus, gate ON, taker 0.10% + funding
- [ ] 2. Sweep: entry-type subsets x volume-filter-all-types x min-momentum x daily-EMA
      confluence x stop widths. Winner = positive in BOTH train and validation at taker.
- [ ] 3. Apply winning config: config.py TIMEFRAME=4h + params, strategy entry filters
- [ ] 4. Tests + byte-compile + update this file with results
- [ ] 5. User restarts bot when flat (running process keeps old code until then)

## Rework results (DONE 2026-06-11)
backtest_entry_filters.py — walk-forward sweep on 4h, gate ON, taker 0.10% + funding 0.005%/4h:
216 configs; 27 positive in BOTH train(2022-24) and validation(2025-26/02). The winning
region is broad, not knife-edge. Two load-bearing changes:
  1. DAILY EMA20 CONFLUENCE (longs above / shorts below, lagged 1 day) — in ALL 27
     survivors; baseline without it: -18R validation.
  2. STOPS 2.0/4.0 ATR (1.2/3.6 in ZERO survivors; 2.5/5.0 also positive = robust).
Chosen config (all 3 entry types kept, vol/mom filters unchanged):
  taker: +17.8R / 608 trades / 45.2% WR / 4.1yr; per-year -15.3(2022) +12.3 +9.4 +0.2 +11.2
  maker ceiling +41.8R; worst-case 0.14% rt still +5.8R. ~3 trades/week.
  Caveat: TC entries carry the edge (+23.8R/583); SQ/MA negligible (25 trades, -6R noise).
  Caveat: 2025 flat (+0.2R); validation profit concentrated in Jan 2026. Edge is real but
  modest (~0.03R/trade avg) — keep on PAPER until live paper confirms.

Shipped:
- config.py: TIMEFRAME 4h, SQZ_SL_ATR_MULT 2.0, SQZ_TP_ATR_MULT 4.0, HTF_CONFLUENCE_ENABLED
- strategy_sqzmom_smc.py: daily_ema() (closed days only, fail-open) + FILTER 1B; defaults 2.0/4.0
- bot.py: passes new params; fetch limit 250->400 (66 days = daily-EMA warmup)
- test_htf_confluence.py 6/6; test_bot_fixes.py de-hardcoded 5m grid, 11/11; gate 8/8; equivalence PASS
- Smoke: real strategy over real 4h windows generates valid signals (RR per type correct)

Investigation verdict (user report "much worse after updates"): NO new bug. The 06-10
partial-candle fix unmasked the 5m strategy's true negative expectancy (26% WR @ 2.1:1
= -0.19R/trade incl fees). It was bleeding before the fix too (Jun 1-9: -$397).

RESTART REQUIRED: running process still has old 5m code. Note: the open paper position's
5m-sized SL/TP will be evaluated against 4h wicks after restart -> likely immediate exit
(one-time ~1R noise). Ideally restart when flat.

## 2026-06-11 (later) — USER ROLLBACK to 5m
User chose to roll back the 4h config and keep old 5m code running. Done via config flip
only (TIMEFRAME=5m, mults 1.2/3.6, HTF_CONFLUENCE_ENABLED=False) — all 4h code paths and
tests kept, restore = flip 4 config values. Bug fixes + gate retained. Running process was
never restarted onto 4h, so disk now matches live behavior; NO restart needed.
User asked for "+$100 per win" (longer profit runs). Backtested TP stretch on 5m at taker:
TP 12x ATR gives exactly +$99/win but win% drops to 12.3%, still -$168/day expectancy
(every TP variant negative: 3.6x/-439, 7.2x/-259, 12x/-168, 24x/-84 $/day, pre-cap).
NOT applied — awaiting explicit user confirmation since it worsens losses.
