"""Tests for the daily-EMA20 confluence filter (strategy_sqzmom_smc)."""
import numpy as np
import pandas as pd

from strategy_sqzmom_smc import SqzMomSmcStrategy

PASSED = 0


def check(label, cond):
    global PASSED
    assert cond, f"FAIL: {label}"
    PASSED += 1
    print(f"[OK ] {label}")


def make_uptrend_df(n=120):
    """4h candles in a steady uptrend (positive momentum, price above EMA34)."""
    idx = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    base = 100.0 + np.arange(n) * 0.5
    wiggle = np.sin(np.arange(n) / 3.0) * 0.2
    close = base + wiggle
    return pd.DataFrame({
        "open": close - 0.1,
        "high": close + 0.3,
        "low": close - 0.3,
        "close": close,
        "volume": np.full(n, 10.0),
    }, index=idx)


# --- 1. daily_ema math ------------------------------------------------------
strat = SqzMomSmcStrategy()
df = make_uptrend_df()

daily_close = df["close"].resample("1D").last()
expected = daily_close.ewm(span=20, adjust=False).mean().iloc[-2]  # through YESTERDAY
got = strat.daily_ema(df)
check("daily_ema = EMA20 of daily closes through yesterday (no lookahead)",
      got is not None and abs(got - expected) < 1e-9)

# --- 2. fail-open cases -----------------------------------------------------
check("non-datetime index fails open (None)",
      strat.daily_ema(df.reset_index(drop=True)) is None)
check("under 2 daily buckets fails open (None)",
      strat.daily_ema(df.iloc[:3]) is None)

# --- 3. filter behavior on a live-like LONG signal --------------------------
allow = SqzMomSmcStrategy(htf_confluence=True)
allow.daily_ema = lambda d, span=20: 0.0  # daily EMA far below price -> aligned
sig = allow.analyze(df, account_balance=1000.0)
check("LONG allowed when price above daily EMA", sig is not None and sig.signal.value == "LONG")

block = SqzMomSmcStrategy(htf_confluence=True)
block.daily_ema = lambda d, span=20: 1e9  # daily EMA far above price -> opposed
check("LONG blocked when price below daily EMA",
      block.analyze(df, account_balance=1000.0) is None)

off = SqzMomSmcStrategy(htf_confluence=False)
off.daily_ema = lambda d, span=20: 1e9  # would block, but filter disabled
check("filter disabled -> signal passes regardless",
      off.analyze(df, account_balance=1000.0) is not None)

print(f"\nALL {PASSED} TESTS PASSED")
