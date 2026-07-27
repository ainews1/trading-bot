"""
Delta-neutral funding-rate arbitrage backtest (spot long + 1x perp short).

Position collects the funding rate every 8h while funding is positive (longs pay
shorts) and PAYS it while negative. No price prediction involved — the only
decisions are when to be in the position and the fee cost of rotating in/out.

Data: real Binance USDT-perp funding history (fapi/v1/fundingRate), cached locally.

Variants:
  - ALWAYS-ON: hold the position the whole time.
  - GATED(enter, exit): enter when trailing-N-avg funding > enter threshold,
    leave when it drops below exit threshold. Each full rotation costs
    ROUND_TRIP fees (spot buy+sell 0.10% x2 + perp open+close 0.05% x2 = 0.30%).

Yields are quoted on SPOT NOTIONAL. Real capital ≈ 1.2-1.5x notional (perp margin),
so divide the APR accordingly for return-on-equity.
"""
import os
import time

import numpy as np
import pandas as pd
import requests

CACHE = "funding_{sym}.csv"
START_MS = int(pd.Timestamp("2022-01-01", tz="UTC").timestamp() * 1000)
ROUND_TRIP = 0.0030     # full enter+exit cost on notional (both legs, taker)
TRAIL_N = 6             # trailing window (6 x 8h = 2 days) for the gate signal


def fetch_funding(sym):
    path = CACHE.format(sym=sym)
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["time"]).set_index("time")
    rows, start = [], START_MS
    while True:
        r = requests.get("https://fapi.binance.com/fapi/v1/fundingRate",
                         params={"symbol": sym, "startTime": start, "limit": 1000},
                         timeout=20)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        start = batch[-1]["fundingTime"] + 1
        if len(batch) < 1000:
            break
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["rate"] = df["fundingRate"].astype(float)
    out = df[["time", "rate"]].set_index("time")
    out.to_csv(path)
    return out


def simulate(rates: pd.Series, enter=None, exit_=None):
    """Return (cum_return_series, n_rotations, frac_time_in_position)."""
    if enter is None:                       # always-on
        pnl = rates.copy()
        return pnl.cumsum(), 1, 1.0
    trail = rates.rolling(TRAIL_N).mean().shift(1)  # decided BEFORE the period accrues
    in_pos = False
    pnl = np.zeros(len(rates))
    rotations = 0
    periods_in = 0
    vals = rates.to_numpy(); tr = trail.to_numpy()
    for i in range(len(vals)):
        if not in_pos and np.isfinite(tr[i]) and tr[i] > enter:
            in_pos = True
            pnl[i] -= ROUND_TRIP / 2          # entry half of the round trip
        elif in_pos and np.isfinite(tr[i]) and tr[i] < exit_:
            in_pos = False
            pnl[i] -= ROUND_TRIP / 2
            rotations += 1
        if in_pos:
            pnl[i] += vals[i]
            periods_in += 1
    return pd.Series(pnl, index=rates.index).cumsum(), rotations, periods_in / len(vals)


def report(sym):
    f = fetch_funding(sym)
    rates = f["rate"]
    yrs = (rates.index[-1] - rates.index[0]).days / 365.25
    print(f"\n=== {sym} ({rates.index[0].date()} .. {rates.index[-1].date()}, {len(rates)} periods) ===")
    print(f"funding positive {(rates > 0).mean()*100:.0f}% of periods | mean {rates.mean()*3*365*100:+.1f}% APR if always-on")

    scenarios = [("ALWAYS-ON", None, None),
                 ("GATED >0", 0.0, 0.0),
                 ("GATED >0.003%/8h", 0.00003, 0.0),
                 ("GATED >0.01%/8h", 0.0001, 0.0)]
    print(f"{'variant':>18} {'totalRet':>9} {'APR':>7} {'maxDD':>7} {'in-pos':>7} {'rotations':>9}")
    for name, en, ex in scenarios:
        cum, rot, frac = simulate(rates, en, ex)
        dd = (cum - cum.cummax()).min()
        print(f"{name:>18} {cum.iloc[-1]*100:>+8.1f}% {cum.iloc[-1]/yrs*100:>+6.1f}% "
              f"{dd*100:>6.2f}% {frac*100:>6.0f}% {rot:>9}")

    # per-year, always-on (the honest baseline)
    cum, _, _ = simulate(rates)
    per_year = rates.groupby(rates.index.year).sum() * 100
    print("per-year always-on return on notional: " +
          "  ".join(f"{y}: {v:+.1f}%" for y, v in per_year.items()))


if __name__ == "__main__":
    for sym in ["BTCUSDT", "ETHUSDT"]:
        report(sym)
    print("\nNote: returns on SPOT NOTIONAL; divide by ~1.3 for return on total capital")
    print("(perp margin). Excludes spot<->perp basis drift (small, mean-reverting) and")
    print("assumes taker fees on rotation (0.30% round trip both legs).")
