"""
Entry-filter walk-forward experiment (follow-up to backtest_variants.py).

backtest_variants.py established: 4h + regime gate removes the fee catastrophe but is only
break-even at taker fees (+5R / 3.5yr). This script searches for an ENTRY-QUALITY filter
that lifts the 4h edge clearly positive at realistic costs, with walk-forward discipline:

  TRAIN    2022-01-01 .. 2024-12-31  (filter selection)
  VALIDATE 2025-01-01 .. 2026-02-05  (untouched by selection; must ALSO be positive)

Engine = the validated logic from backtest_sqzmom_gate.py (2066/2066 signals matched),
extended with:
  - funding cost while a position is held (perp funding ~0.01%/8h on notional)
  - sweepable entry filters: entry-type subset, volume filter on ALL types,
    minimum momentum strength (ATR units), daily-EMA20 confluence (no lookahead)

Costs: taker 0.10% round-trip on notional (realistic market orders) + funding.
Gate is always ON (proven strict improvement in every prior variant).
"""
import numpy as np, pandas as pd
from itertools import product

RAW = pd.read_csv("historical_data.csv", parse_dates=["timestamp"]).set_index("timestamp")

TRAIN_END = pd.Timestamp("2025-01-01")
TAKER = 0.0010            # round-trip on notional
FUNDING_8H = 0.0001       # 0.01% per 8h on notional while held

# Fixed indicator params (same as live strategy / validated engine)
BB_LEN, BB_MULT, KC_LEN, KC_MULT, MOM_LEN = 16, 1.8, 16, 1.0, 8
EMA_TREND, EMA_TOL, ATR_PERIOD = 34, 0.001, 10
RISK = 20.0
MAX_CONSEC, COOLDOWN = 4, 3


def resample(tf):
    if tf == "5min":
        return RAW
    return RAW.resample(tf).agg({"open": "first", "high": "max", "low": "min",
                                 "close": "last", "volume": "sum"}).dropna()


def precompute(df):
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    n = len(df)
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr_kc = tr.rolling(KC_LEN).mean()
    bb_mid = close.rolling(BB_LEN).mean(); bb_std = close.rolling(BB_LEN).std()
    bb_up, bb_lo = bb_mid + BB_MULT * bb_std, bb_mid - BB_MULT * bb_std
    kc_mid = close.ewm(span=KC_LEN, adjust=False).mean()
    kc_up, kc_lo = kc_mid + KC_MULT * atr_kc, kc_mid - KC_MULT * atr_kc
    sqz_on = ((bb_lo > kc_lo) & (bb_up < kc_up)).to_numpy()
    midline = (kc_mid + bb_mid) / 2; delta = (close - midline).to_numpy()
    x = np.arange(MOM_LEN)
    w = np.array([np.polyval(np.polyfit(x, e, 1), MOM_LEN - 1) for e in np.eye(MOM_LEN)])
    mom = np.full(n, np.nan)
    mom[MOM_LEN - 1:] = np.convolve(delta, w[::-1], mode="valid")
    atr = tr.rolling(ATR_PERIOD).mean().to_numpy()
    ema_trend = close.ewm(span=EMA_TREND, adjust=False).mean().to_numpy()
    vol_avg = vol.rolling(20).mean().to_numpy()
    ema9 = close.ewm(span=9, adjust=False).mean().to_numpy()
    ema21 = close.ewm(span=21, adjust=False).mean().to_numpy()
    ema20 = close.ewm(span=20, adjust=False).mean().to_numpy()
    cl = close.to_numpy()
    trend = np.where((ema9 > ema21) & (cl > ema9), "BULL",
            np.where((ema9 < ema21) & (cl < ema9), "BEAR", "NEUTRAL"))
    strength = np.where(trend == "BULL", np.minimum(1.0, (ema9 - ema21) / ema21 * 100),
               np.where(trend == "BEAR", np.minimum(1.0, (ema21 - ema9) / ema9 * 100), 0.0))
    rng20 = ((high.rolling(20).max() - low.rolling(20).min()) / close).to_numpy()
    rng50 = ((high.rolling(50).max() - low.rolling(50).min()) / close).to_numpy()
    regime = np.where(strength > 0.3, "TRENDING", np.where(rng20 < rng50 * 0.5, "RANGING",
             np.where(np.abs(cl - ema20) / cl > 0.02, "BREAKOUT", "REVERSAL")))

    # Daily EMA20 confluence — shift(1) so each 4h bar only sees FULLY CLOSED days
    daily_close = df["close"].resample("1D").last()
    daily_ema = daily_close.ewm(span=20, adjust=False).mean().shift(1)
    htf_ema = daily_ema.reindex(df.index, method="ffill").to_numpy()

    return dict(n=n, cl=cl, hi=high.to_numpy(), lo=low.to_numpy(), vol=vol.to_numpy(),
                sqz_on=sqz_on, mom=mom, atr=atr, ema_trend=ema_trend, vol_avg=vol_avg,
                trend=trend, regime=regime, htf_ema=htf_ema, index=df.index)


def backtest(d, sl_mult, tp_mult, types, vol_all, min_mom, htf_on,
             cost=TAKER, funding_per_candle=0.0):
    n, cl, hi, lo, volp = d["n"], d["cl"], d["hi"], d["lo"], d["vol"]
    sqz_on, mom, atr, ema_trend, vol_avg = d["sqz_on"], d["mom"], d["atr"], d["ema_trend"], d["vol_avg"]
    trend, regime, htf_ema, index = d["trend"], d["regime"], d["htf_ema"], d["index"]

    def signal_at(i):
        a, m, mp = atr[i], mom[i], mom[i - 1]
        if not np.isfinite(a) or a == 0 or not np.isfinite(m) or not np.isfinite(mp):
            return None
        if np.isnan(ema_trend[i]) or np.isnan(vol_avg[i]):
            return None
        price = cl[i]; fired = sqz_on[i - 1] and not sqz_on[i]; on = sqz_on[i]
        m2 = mom[i - 2] if i >= 2 else np.nan; m3 = mom[i - 3] if i >= 3 else np.nan
        direction = stype = None
        if fired and "SQ" in types:
            direction = "LONG" if m > 0 else ("SHORT" if m < 0 else None); stype = "SQ"
        elif on and "MA" in types and not np.isnan(mp) and not np.isnan(m2):
            accel, accel_p = m - mp, mp - m2; mss = a * 0.08
            if m > mss and accel > 0 and accel_p > 0: direction, stype = "LONG", "MA"
            elif m < -mss and accel < 0 and accel_p < 0: direction, stype = "SHORT", "MA"
        if stype is None and "TC" in types and not np.isnan(m2) and not np.isnan(m3):
            mtm = a * 0.20
            if m > 0 and mp > 0 and m2 > 0 and m > mtm and price > ema_trend[i]:
                direction, stype = "LONG", "TC"
            elif m < 0 and mp < 0 and m2 < 0 and abs(m) > mtm and price < ema_trend[i]:
                direction, stype = "SHORT", "TC"
        if stype is None or direction is None:
            return None
        tol = ema_trend[i] * EMA_TOL
        if direction == "LONG" and price < ema_trend[i] - tol: return None
        if direction == "SHORT" and price > ema_trend[i] + tol: return None
        # min momentum strength — ALL entry types (was 0.01 on SQ only)
        if abs(m) < a * min_mom: return None
        # volume confirmation — scope configurable
        vmult = vol_all if vol_all is not None else (0.8 if stype == "SQ" else None)
        if vmult is not None and vol_avg[i] > 0 and volp[i] < vol_avg[i] * vmult:
            return None
        # daily EMA confluence
        if htf_on and not np.isnan(htf_ema[i]):
            if direction == "LONG" and price < htf_ema[i]: return None
            if direction == "SHORT" and price > htf_ema[i]: return None
        if stype == "SQ": sm, tm = sl_mult, tp_mult
        elif stype == "MA": sm, tm = sl_mult * 0.85, tp_mult * 0.67
        else: sm, tm = sl_mult * 0.7, tp_mult * 0.50
        return direction, sm, tm, stype

    def blocked(i, direction):
        if regime[i] == "RANGING": return True
        if direction == "LONG" and trend[i] == "BEAR": return True
        if direction == "SHORT" and trend[i] == "BULL": return True
        return False

    pos = None; cooldown = consec = 0; trades = []
    start = max(EMA_TREND + 10, 50, KC_LEN)
    for i in range(start, n):
        if pos is not None:
            ex = None
            if pos["side"] == "LONG":
                if lo[i] <= pos["sl"]: ex = pos["sl"]
                elif hi[i] >= pos["tp"]: ex = pos["tp"]
            else:
                if hi[i] >= pos["sl"]: ex = pos["sl"]
                elif lo[i] <= pos["tp"]: ex = pos["tp"]
            if ex is not None:
                diff = (ex - pos["entry"]) if pos["side"] == "LONG" else (pos["entry"] - ex)
                gross = pos["size"] * diff
                notional = pos["size"] * pos["entry"]
                fee = notional * cost + notional * funding_per_candle * (i - pos["i"])
                pnl = gross - fee; won = pnl > 0
                consec = 0 if won else consec + 1
                if consec >= MAX_CONSEC: cooldown = COOLDOWN; consec = 0
                trades.append({"ts": index[i], "pnl": pnl, "won": won,
                               "R": pnl / RISK, "type": pos["type"]})
                pos = None
            continue
        if cooldown > 0: cooldown -= 1; continue
        sig = signal_at(i)
        if sig is None: continue
        direction, sm, tm, stype = sig
        if blocked(i, direction): continue
        price = cl[i]; a = atr[i]
        if direction == "LONG": sl, tp = price - a * sm, price + a * tm
        else: sl, tp = price + a * sm, price - a * tm
        size = RISK / abs(price - sl)
        pos = {"side": direction, "entry": price, "sl": sl, "tp": tp,
               "size": size, "type": stype, "i": i}
    return trades


def split_summ(trades):
    def s(tr):
        if not tr: return (0, 0.0, 0.0)
        R = np.array([t["R"] for t in tr])
        return len(R), (R > 0).mean() * 100, R.sum()
    train = [t for t in trades if t["ts"] < TRAIN_END]
    val = [t for t in trades if t["ts"] >= TRAIN_END]
    return s(train), s(val)


if __name__ == "__main__":
    df4 = resample("4h")
    d = precompute(df4)
    funding_4h = FUNDING_8H / 2  # per 4h candle

    TYPE_SETS = [("SQ",), ("SQ", "MA"), ("SQ", "TC"), ("SQ", "MA", "TC")]
    VOL_ALL = [None, 0.8, 1.0]          # None = current behavior (0.8 on SQ only)
    MIN_MOM = [0.01, 0.05, 0.10]
    HTF = [False, True]
    STOPS = [(1.2, 3.6), (2.0, 4.0), (2.5, 5.0)]

    rows = []
    for types, vol_all, min_mom, htf_on, (sl, tp) in product(
            TYPE_SETS, VOL_ALL, MIN_MOM, HTF, STOPS):
        trades = backtest(d, sl, tp, set(types), vol_all, min_mom, htf_on,
                          cost=TAKER, funding_per_candle=funding_4h)
        (tn, twr, tR), (vn, vwr, vR) = split_summ(trades)
        rows.append(dict(types="+".join(types), vol=str(vol_all), mom=min_mom,
                         htf=htf_on, stops=f"{sl}/{tp}",
                         tn=tn, twr=twr, tR=tR, vn=vn, vwr=vwr, vR=vR,
                         avg=(tR + vR) / max(tn + vn, 1)))

    res = pd.DataFrame(rows)
    # Baseline = current live behavior on 4h (all types, vol on SQ only, mom 0.01)
    base = res[(res.types == "SQ+MA+TC") & (res.vol == "None") &
               (res.mom == 0.01) & (~res.htf) & (res.stops == "1.2/3.6")]
    print("BASELINE (current logic on 4h, taker+funding):")
    print(base.to_string(index=False), "\n")

    ok = res[(res.tR > 0) & (res.vR > 0) & (res.vn >= 25)]
    print(f"Configs positive in BOTH periods with >=25 validation trades: {len(ok)}/{len(res)}")
    print(ok.sort_values("vR", ascending=False).head(25).to_string(index=False))
    print("\nTop 10 by combined avg R/trade (min 25 val trades):")
    print(ok.sort_values("avg", ascending=False).head(10).to_string(index=False))
