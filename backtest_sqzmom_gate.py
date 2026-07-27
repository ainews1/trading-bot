"""
Out-of-sample, fee-aware backtest of SQZMOM_SMC with vs without the regime gate.

Data: historical_data.csv (5m, 2022-01 -> 2026-02-05) — ends BEFORE the paper-
trading period the gate rule was derived from, so this is genuine out-of-sample.

Faithful to strategy_sqzmom_smc.py entry logic (SQUEEZE_FIRE / MOM_ACCEL / TREND_CONT,
EMA34 alignment, volume + momentum-strength filters, ATR SL/TP, cooldown, 2% risk).
Omits the loose SMC-bias gate (the slow part); both arms omit it equally so the
gate-on vs gate-off DELTA — the thing under test — remains valid.

Gate logic mirrors bot.py::_regime_gate (block RANGING regime + counter-trend entries).
"""
import numpy as np, pandas as pd

# ---- live config (config.py SQZ_* + strategy defaults) ----
BB_LEN, BB_MULT, KC_LEN, KC_MULT, MOM_LEN = 16, 1.8, 16, 1.0, 8
EMA_TREND, EMA_TOL = 34, 0.001
ATR_PERIOD = 10
SL_MULT, TP_MULT = 1.2, 3.6
RISK = 0.02
VOL_CONFIRM = 0.8
MAX_CONSEC_LOSS, COOLDOWN = 4, 3
START_BAL = 1000.0
# fees: Poloniex perp taker ~0.05% + ~0.02% slippage, round trip on notional
COST_RT = 2 * (0.0005 + 0.0002)   # 0.14% of notional per round trip

df = pd.read_csv("historical_data.csv", parse_dates=["timestamp"]).set_index("timestamp")
close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
n = len(df)

# ---- squeeze indicators (vectorized) ----
tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
atr_kc = tr.rolling(KC_LEN).mean()
bb_mid = close.rolling(BB_LEN).mean()
bb_std = close.rolling(BB_LEN).std()
bb_up, bb_lo = bb_mid + BB_MULT * bb_std, bb_mid - BB_MULT * bb_std
kc_mid = close.ewm(span=KC_LEN, adjust=False).mean()
kc_up, kc_lo = kc_mid + KC_MULT * atr_kc, kc_mid - KC_MULT * atr_kc
sqz_on = ((bb_lo > kc_lo) & (bb_up < kc_up)).to_numpy()

# linreg-endpoint momentum via fixed weight vector (exact, no per-bar polyfit)
midline = (kc_mid + bb_mid) / 2
delta = (close - midline).to_numpy()
x = np.arange(MOM_LEN)
w = np.array([np.polyval(np.polyfit(x, e, 1), MOM_LEN - 1)
              for e in np.eye(MOM_LEN)])          # weight per lag
mom = np.full(n, np.nan)
conv = np.convolve(delta, w[::-1], mode="valid")  # aligned to end of each window
mom[MOM_LEN - 1:] = conv

atr = tr.rolling(ATR_PERIOD).mean().to_numpy()
ema_trend = close.ewm(span=EMA_TREND, adjust=False).mean().to_numpy()
vol_avg = vol.rolling(20).mean().to_numpy()
closep, highp, lowp, volp = close.to_numpy(), high.to_numpy(), low.to_numpy(), vol.to_numpy()

# ---- scout trend + regime (vectorized, mirrors market_scout.py) ----
ema9 = close.ewm(span=9, adjust=False).mean().to_numpy()
ema21 = close.ewm(span=21, adjust=False).mean().to_numpy()
ema20 = close.ewm(span=20, adjust=False).mean().to_numpy()
trend_dir = np.where((ema9 > ema21) & (closep > ema9), "BULL",
            np.where((ema9 < ema21) & (closep < ema9), "BEAR", "NEUTRAL"))
strength = np.where(trend_dir == "BULL", np.minimum(1.0, (ema9 - ema21) / ema21 * 100),
            np.where(trend_dir == "BEAR", np.minimum(1.0, (ema21 - ema9) / ema9 * 100), 0.0))
rng20 = ((high.rolling(20).max() - low.rolling(20).min()) / close).to_numpy()
rng50 = ((high.rolling(50).max() - low.rolling(50).min()) / close).to_numpy()
regime = np.where(strength > 0.3, "TRENDING",
         np.where(rng20 < rng50 * 0.5, "RANGING",
         np.where(np.abs(closep - ema20) / closep > 0.02, "BREAKOUT", "REVERSAL")))

def signal_at(i):
    """Return (direction, sl_mult, tp_mult) or None — mirrors analyze() minus SMC gate."""
    a, m, mp = atr[i], mom[i], mom[i - 1]
    if not np.isfinite(a) or a == 0 or not np.isfinite(m) or not np.isfinite(mp):
        return None
    if np.isnan(ema_trend[i]) or np.isnan(vol_avg[i]):
        return None
    price = closep[i]
    fired = sqz_on[i - 1] and not sqz_on[i]
    on = sqz_on[i]
    m2 = mom[i - 2] if i >= 2 else np.nan
    m3 = mom[i - 3] if i >= 3 else np.nan
    direction = stype = None

    if fired:
        direction = "LONG" if m > 0 else ("SHORT" if m < 0 else None)
        stype = "SQ"
    elif on and not np.isnan(mp) and not np.isnan(m2):
        accel, accel_p = m - mp, mp - m2
        mss = a * 0.08
        if m > mss and accel > 0 and accel_p > 0:
            direction, stype = "LONG", "MA"
        elif m < -mss and accel < 0 and accel_p < 0:
            direction, stype = "SHORT", "MA"
    if stype is None and not np.isnan(m2) and not np.isnan(m3):
        mtm = a * 0.20
        if m > 0 and mp > 0 and m2 > 0 and m > mtm and price > ema_trend[i]:
            direction, stype = "LONG", "TC"
        elif m < 0 and mp < 0 and m2 < 0 and abs(m) > mtm and price < ema_trend[i]:
            direction, stype = "SHORT", "TC"
    if stype is None or direction is None:
        return None

    # EMA34 alignment
    tol = ema_trend[i] * EMA_TOL
    if direction == "LONG" and price < ema_trend[i] - tol:
        return None
    if direction == "SHORT" and price > ema_trend[i] + tol:
        return None
    # volume + momentum-strength (squeeze fire only)
    if stype == "SQ":
        if vol_avg[i] > 0 and volp[i] < vol_avg[i] * VOL_CONFIRM:
            return None
        if abs(m) < a * 0.01:
            return None
    if stype == "SQ":   sm, tm = SL_MULT, TP_MULT
    elif stype == "MA": sm, tm = SL_MULT * 0.85, TP_MULT * 0.67
    else:               sm, tm = SL_MULT * 0.7, TP_MULT * 0.50
    return direction, sm, tm

def gate_blocks(i, direction):
    if regime[i] == "RANGING":
        return True
    if direction == "LONG" and trend_dir[i] == "BEAR":
        return True
    if direction == "SHORT" and trend_dir[i] == "BULL":
        return True
    return False

RISK_DOLLARS = START_BAL * RISK   # fixed $ risk/trade — isolates per-trade edge, no ruin spiral

def run(gate_on, fees, cost=COST_RT):
    bal = START_BAL
    pos = None          # dict or None
    cooldown = consec = 0
    trades = []
    start = max(EMA_TREND + 10, 50, KC_LEN, 50)
    for i in range(start, n):
        if pos is not None:
            # exit check on this candle (SL priority, mirrors check_paper_exit)
            exit_px = reason = None
            if pos["side"] == "LONG":
                if lowp[i] <= pos["sl"]: exit_px, reason = pos["sl"], "SL"
                elif highp[i] >= pos["tp"]: exit_px, reason = pos["tp"], "TP"
            else:
                if highp[i] >= pos["sl"]: exit_px, reason = pos["sl"], "SL"
                elif lowp[i] <= pos["tp"]: exit_px, reason = pos["tp"], "TP"
            if exit_px is not None:
                diff = (exit_px - pos["entry"]) if pos["side"] == "LONG" else (pos["entry"] - exit_px)
                gross = pos["size"] * diff
                fee = pos["size"] * pos["entry"] * cost if fees else 0.0
                pnl = gross - fee
                bal += pnl
                won = pnl > 0
                consec = 0 if won else consec + 1
                if consec >= MAX_CONSEC_LOSS:
                    cooldown = COOLDOWN; consec = 0
                trades.append({"ts": df.index[i], "pnl": pnl, "won": won,
                               "side": pos["side"], "regime": pos["regime"],
                               "trend": pos["trend"], "bal": bal})
                pos = None
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        sig = signal_at(i)
        if sig is None:
            continue
        direction, sm, tm = sig
        if gate_on and gate_blocks(i, direction):
            continue
        price = closep[i]; a = atr[i]
        if direction == "LONG":
            sl, tp = price - a * sm, price + a * tm
        else:
            sl, tp = price + a * sm, price - a * tm
        size = RISK_DOLLARS / abs(price - sl)   # fixed-risk: loss at SL == RISK_DOLLARS
        pos = {"side": direction, "entry": price, "sl": sl, "tp": tp, "size": size,
               "regime": regime[i], "trend": trend_dir[i]}
    return bal, trades

def stats(label, bal, trades):
    if not trades:
        print(f"{label:28} no trades"); return
    pnls = np.array([t["pnl"] for t in trades])
    wins = pnls[pnls > 0]; losses = pnls[pnls <= 0]
    wr = len(wins) / len(pnls) * 100
    pf = (wins.sum() / -losses.sum()) if losses.size and losses.sum() < 0 else float("inf")
    total = pnls.sum()
    avg_R = np.mean(pnls) / RISK_DOLLARS   # expectancy in risk-multiples
    print(f"{label:28} n={len(pnls):>5} win%={wr:5.1f} totalPnL=${total:+9.0f} "
          f"exp=${np.mean(pnls):+6.2f} ({avg_R:+.3f}R) PF={pf:4.2f}")

def main():
    print(f"Candles: {n}  span {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"Fees: round-trip {COST_RT*100:.2f}% of notional | risk ${RISK_DOLLARS:.0f}/trade (fixed)\n")

    results = {}
    for gate in (False, True):
        for fees in (False, True):
            bal, trades = run(gate, fees)
            results[(gate, fees)] = (bal, trades)
            tag = ("GATE ON " if gate else "GATE OFF") + (" net" if fees else " gross")
            stats(tag, bal, trades)

    print("\n=== Per-year (NET of fees) ===")
    for gate in (False, True):
        _, trades = results[(gate, True)]
        by_year = {}
        for t in trades:
            by_year.setdefault(t["ts"].year, []).append(t)
        print(f"  {'GATE ON' if gate else 'GATE OFF'}:")
        for y in sorted(by_year):
            g = by_year[y]
            wr = sum(t['won'] for t in g) / len(g) * 100
            print(f"    {y}: n={len(g):>4} win%={wr:5.1f} totalPnL=${sum(t['pnl'] for t in g):+9.0f}")

    # ---- fee sensitivity (GATE ON): at what round-trip cost does it break even? ----
    print("\n=== Fee sensitivity (GATE ON) — expectancy vs round-trip cost ===")
    for rt in (0.0000, 0.0002, 0.0005, 0.0007, 0.0010, 0.0014):
        _, tr_s = run(True, True, cost=rt)
        pnls = np.array([t["pnl"] for t in tr_s])
        print(f"    round-trip={rt*100:.2f}%  exp=${pnls.mean():+7.2f} "
              f"({pnls.mean()/RISK_DOLLARS:+.3f}R) win%={(pnls>0).mean()*100:4.1f} "
              f"total=${pnls.sum():+11.0f}")

    print("\n=== Edge-decay (GATE ON, net) — quarterly win% & expectancy ===")
    _, trades = results[(True, True)]
    buckets = {}
    for t in trades:
        q = f"{t['ts'].year}Q{(t['ts'].month-1)//3+1}"
        buckets.setdefault(q, []).append(t["pnl"])
    for q in sorted(buckets):
        p = buckets[q]
        wr = sum(1 for x in p if x > 0) / len(p) * 100
        print(f"    {q}: n={len(p):>3} win%={wr:5.1f} exp=${np.mean(p):+6.2f} total=${sum(p):+8.0f}")

if __name__ == "__main__":
    main()
