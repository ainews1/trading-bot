"""Validate backtest_sqzmom_gate.signal_at() against the REAL strategy.analyze().
Real signals (minus the omitted SMC gate) must be a subset of the reimplementation,
with matching direction."""
import numpy as np, pandas as pd
from strategy_sqzmom_smc import SqzMomSmcStrategy
import backtest_sqzmom_gate as bt

df = bt.df
strat = SqzMomSmcStrategy(bb_length=16, bb_mult=1.8, kc_length=16, kc_mult=1.0,
                          mom_length=8, risk_per_trade=0.02, leverage=10)

LO, HI, WIN = 400000, 403000, 300
real, mine = {}, {}
for i in range(LO, HI):
    sig = strat.analyze(df.iloc[i - WIN:i + 1], 1000.0)
    if sig is not None and sig.signal.value in ("LONG", "SHORT"):
        real[i] = sig.signal.value
    m = bt.signal_at(i)
    if m is not None:
        mine[i] = m[0]

agree = sum(1 for i in real if i in mine and mine[i] == real[i])
real_not_mine = [i for i in real if i not in mine]          # mine MISSING a real signal (bad)
dir_conflict = [i for i in real if i in mine and mine[i] != real[i]]  # opposite dir (bad)
mine_extra = [i for i in mine if i not in real]             # mine has, real blocked (SMC) — expected

print(f"window {LO}-{HI} ({HI-LO} candles)")
print(f"real signals: {len(real)} | mine signals: {len(mine)}")
print(f"  matched (same dir):       {agree}")
print(f"  real-but-not-mine (BAD):  {len(real_not_mine)}  {real_not_mine[:10]}")
print(f"  direction conflicts(BAD): {len(dir_conflict)}  {dir_conflict[:10]}")
print(f"  mine-extra (SMC-blocked, expected): {len(mine_extra)}")
ok = len(real_not_mine) == 0 and len(dir_conflict) == 0
print("VALIDATION", "PASS — reimplementation reproduces real entry logic" if ok
      else "FAIL — reimplementation diverges from real strategy")
