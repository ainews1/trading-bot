"""
Equivalence test: vectorized calculate_squeeze() vs the original implementation
(per-bar np.polyfit momentum loop + pd.concat True Range).

Run: python test_squeeze_equivalence.py
"""

import time

import numpy as np
import pandas as pd

from strategy_sqzmom_smc import SqzMomSmcStrategy


def reference_squeeze(df, bb_length, bb_mult, kc_length, kc_mult, mom_length,
                      atr_period, ema_trend_period):
    """Verbatim copy of the ORIGINAL calculate_squeeze (pre-vectorization)."""
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(kc_length).mean()

    bb_mid = close.rolling(bb_length).mean()
    bb_std = close.rolling(bb_length).std()
    bb_upper = bb_mid + bb_mult * bb_std
    bb_lower = bb_mid - bb_mult * bb_std

    kc_mid = close.ewm(span=kc_length, adjust=False).mean()
    kc_upper = kc_mid + kc_mult * atr
    kc_lower = kc_mid - kc_mult * atr

    df["sqz_on"] = (bb_lower > kc_lower) & (bb_upper < kc_upper)

    midline = (kc_mid + bb_mid) / 2
    delta = close - midline

    mom_values = np.full(len(df), np.nan)
    x = np.arange(mom_length)
    for i in range(mom_length - 1, len(df)):
        y = delta.iloc[i - mom_length + 1: i + 1].values
        if np.any(np.isnan(y)):
            continue
        coeffs = np.polyfit(x, y, 1)
        mom_values[i] = np.polyval(coeffs, mom_length - 1)

    df["sqz_mom"] = mom_values
    df["sqz_mom_prev"] = pd.Series(mom_values).shift(1).values
    df["atr"] = tr.rolling(atr_period).mean()
    df["ema_trend"] = close.ewm(span=ema_trend_period, adjust=False).mean()
    df["vol_avg"] = df["volume"].rolling(20).mean()
    return df


def run_case(df, label, **params):
    strat = SqzMomSmcStrategy(
        bb_length=params["bb_length"], bb_mult=params["bb_mult"],
        kc_length=params["kc_length"], kc_mult=params["kc_mult"],
        mom_length=params["mom_length"],
    )
    params["atr_period"] = strat.atr_period
    params["ema_trend_period"] = strat.ema_trend_period

    t0 = time.perf_counter()
    ref = reference_squeeze(df, **params)
    t_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    new = strat.calculate_squeeze(df)
    t_new = time.perf_counter() - t0

    for col in ["sqz_mom", "sqz_mom_prev", "atr", "ema_trend", "vol_avg"]:
        a, b = ref[col].to_numpy(float), new[col].to_numpy(float)
        assert np.allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True), \
            f"{label}: column {col} differs (max diff {np.nanmax(np.abs(a - b))})"
    assert (ref["sqz_on"] == new["sqz_on"]).all(), f"{label}: sqz_on differs"

    print(f"  PASS {label}: {len(df)} rows | ref {t_ref*1000:.1f}ms -> new {t_new*1000:.1f}ms "
          f"({t_ref/max(t_new, 1e-9):.0f}x faster)")


if __name__ == "__main__":
    full = pd.read_csv("historical_data.csv", parse_dates=["timestamp"], index_col="timestamp")

    # Live config params (config.py) and strategy defaults
    live = dict(bb_length=16, bb_mult=1.8, kc_length=16, kc_mult=1.0, mom_length=8)
    defaults = dict(bb_length=20, bb_mult=2.0, kc_length=20, kc_mult=1.5, mom_length=12)

    print("Squeeze equivalence (vectorized vs original polyfit loop):")
    run_case(full.tail(250).copy(), "live-config, 250 rows (bot fetch size)", **live)
    run_case(full.tail(5000).copy(), "live-config, 5000 rows", **live)
    run_case(full.iloc[100_000:105_000].copy(), "live-config, mid-history slice", **live)
    run_case(full.tail(5000).copy(), "default params, 5000 rows", **defaults)
    run_case(full.tail(30).copy(), "tiny frame (30 rows, NaN warmup)", **live)
    run_case(full.tail(5).copy(), "frame shorter than mom_length", **live)
    print("All equivalence checks passed.")
