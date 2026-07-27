"""
Structural-rework experiment: can a lower-frequency / wider-stop / maker-fee variant
of SQZMOM_SMC clear the fee hurdle that kills the 5m version?

Engine is the validated logic from backtest_sqzmom_gate.py (validate_backtest.py: 2066/2066
signals matched), lifted into a parameterized function so we can sweep timeframe, stop width,
fee level, and gate on/off on identical resampled candles.

Maker-fill caveat: a low 'maker' cost assumes limit-order entries actually fill at the
intended price. For breakout-style entries that's optimistic; treat the maker rows as a
best-case ceiling, taker rows as the realistic floor.
"""
import numpy as np, pandas as pd
from dataclasses import dataclass

RAW = pd.read_csv("historical_data.csv", parse_dates=["timestamp"]).set_index("timestamp")

def resample(tf):
    if tf == "5min":
        return RAW
    o = RAW.resample(tf).agg({"open": "first", "high": "max", "low": "min",
                              "close": "last", "volume": "sum"}).dropna()
    return o

@dataclass
class P:
    bb_len=16; bb_mult=1.8; kc_len=16; kc_mult=1.0; mom_len=8
    ema_trend=34; ema_tol=0.001; atr_period=10
    sl_mult=1.2; tp_mult=3.6
    risk=20.0; vol_confirm=0.8; max_consec=4; cooldown=3
    cost=0.0; gate_on=False

def precompute(df, p):
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    n = len(df)
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr_kc = tr.rolling(p.kc_len).mean()
    bb_mid = close.rolling(p.bb_len).mean(); bb_std = close.rolling(p.bb_len).std()
    bb_up, bb_lo = bb_mid+p.bb_mult*bb_std, bb_mid-p.bb_mult*bb_std
    kc_mid = close.ewm(span=p.kc_len, adjust=False).mean()
    kc_up, kc_lo = kc_mid+p.kc_mult*atr_kc, kc_mid-p.kc_mult*atr_kc
    sqz_on = ((bb_lo > kc_lo) & (bb_up < kc_up)).to_numpy()
    midline = (kc_mid+bb_mid)/2; delta = (close-midline).to_numpy()
    x = np.arange(p.mom_len)
    w = np.array([np.polyval(np.polyfit(x, e, 1), p.mom_len-1) for e in np.eye(p.mom_len)])
    mom = np.full(n, np.nan)
    mom[p.mom_len-1:] = np.convolve(delta, w[::-1], mode="valid")
    atr = tr.rolling(p.atr_period).mean().to_numpy()
    ema_trend = close.ewm(span=p.ema_trend, adjust=False).mean().to_numpy()
    vol_avg = vol.rolling(20).mean().to_numpy()
    ema9 = close.ewm(span=9, adjust=False).mean().to_numpy()
    ema21 = close.ewm(span=21, adjust=False).mean().to_numpy()
    ema20 = close.ewm(span=20, adjust=False).mean().to_numpy()
    cl = close.to_numpy()
    trend = np.where((ema9>ema21)&(cl>ema9), "BULL", np.where((ema9<ema21)&(cl<ema9), "BEAR", "NEUTRAL"))
    strength = np.where(trend=="BULL", np.minimum(1.0,(ema9-ema21)/ema21*100),
               np.where(trend=="BEAR", np.minimum(1.0,(ema21-ema9)/ema9*100), 0.0))
    rng20 = ((high.rolling(20).max()-low.rolling(20).min())/close).to_numpy()
    rng50 = ((high.rolling(50).max()-low.rolling(50).min())/close).to_numpy()
    regime = np.where(strength>0.3, "TRENDING", np.where(rng20<rng50*0.5, "RANGING",
             np.where(np.abs(cl-ema20)/cl>0.02, "BREAKOUT", "REVERSAL")))
    return dict(n=n, cl=cl, hi=high.to_numpy(), lo=low.to_numpy(), vol=vol.to_numpy(),
                sqz_on=sqz_on, mom=mom, atr=atr, ema_trend=ema_trend, vol_avg=vol_avg,
                trend=trend, regime=regime)

def backtest(df, p):
    d = precompute(df, p)
    n, cl, hi, lo, volp = d["n"], d["cl"], d["hi"], d["lo"], d["vol"]
    sqz_on, mom, atr, ema_trend, vol_avg = d["sqz_on"], d["mom"], d["atr"], d["ema_trend"], d["vol_avg"]
    trend, regime = d["trend"], d["regime"]

    def signal_at(i):
        a, m, mp = atr[i], mom[i], mom[i-1]
        if not np.isfinite(a) or a == 0 or not np.isfinite(m) or not np.isfinite(mp): return None
        if np.isnan(ema_trend[i]) or np.isnan(vol_avg[i]): return None
        price = cl[i]; fired = sqz_on[i-1] and not sqz_on[i]; on = sqz_on[i]
        m2 = mom[i-2] if i>=2 else np.nan; m3 = mom[i-3] if i>=3 else np.nan
        direction = stype = None
        if fired:
            direction = "LONG" if m>0 else ("SHORT" if m<0 else None); stype="SQ"
        elif on and not np.isnan(mp) and not np.isnan(m2):
            accel, accel_p = m-mp, mp-m2; mss = a*0.08
            if m>mss and accel>0 and accel_p>0: direction, stype = "LONG","MA"
            elif m<-mss and accel<0 and accel_p<0: direction, stype = "SHORT","MA"
        if stype is None and not np.isnan(m2) and not np.isnan(m3):
            mtm = a*0.20
            if m>0 and mp>0 and m2>0 and m>mtm and price>ema_trend[i]: direction, stype="LONG","TC"
            elif m<0 and mp<0 and m2<0 and abs(m)>mtm and price<ema_trend[i]: direction, stype="SHORT","TC"
        if stype is None or direction is None: return None
        tol = ema_trend[i]*p.ema_tol
        if direction=="LONG" and price < ema_trend[i]-tol: return None
        if direction=="SHORT" and price > ema_trend[i]+tol: return None
        if stype=="SQ":
            if vol_avg[i]>0 and volp[i] < vol_avg[i]*p.vol_confirm: return None
            if abs(m) < a*0.01: return None
        if stype=="SQ": sm,tm = p.sl_mult, p.tp_mult
        elif stype=="MA": sm,tm = p.sl_mult*0.85, p.tp_mult*0.67
        else: sm,tm = p.sl_mult*0.7, p.tp_mult*0.50
        return direction, sm, tm

    def blocked(i, direction):
        if regime[i]=="RANGING": return True
        if direction=="LONG" and trend[i]=="BEAR": return True
        if direction=="SHORT" and trend[i]=="BULL": return True
        return False

    pos=None; cooldown=consec=0; trades=[]; notional_r=[]
    start = max(p.ema_trend+10, 50, p.kc_len)
    for i in range(start, n):
        if pos is not None:
            ex=None
            if pos["side"]=="LONG":
                if lo[i]<=pos["sl"]: ex=pos["sl"]
                elif hi[i]>=pos["tp"]: ex=pos["tp"]
            else:
                if hi[i]>=pos["sl"]: ex=pos["sl"]
                elif lo[i]<=pos["tp"]: ex=pos["tp"]
            if ex is not None:
                diff = (ex-pos["entry"]) if pos["side"]=="LONG" else (pos["entry"]-ex)
                gross = pos["size"]*diff
                fee = pos["size"]*pos["entry"]*p.cost
                pnl = gross-fee; won = pnl>0
                consec = 0 if won else consec+1
                if consec>=p.max_consec: cooldown=p.cooldown; consec=0
                trades.append({"ts": df.index[i], "pnl": pnl, "won": won, "R": pnl/p.risk})
                pos=None
            continue
        if cooldown>0: cooldown-=1; continue
        sig = signal_at(i)
        if sig is None: continue
        direction, sm, tm = sig
        if p.gate_on and blocked(i, direction): continue
        price=cl[i]; a=atr[i]
        if direction=="LONG": sl,tp = price-a*sm, price+a*tm
        else: sl,tp = price+a*sm, price-a*tm
        size = p.risk/abs(price-sl)
        notional_r.append(size*price/p.risk)
        pos={"side":direction,"entry":price,"sl":sl,"tp":tp,"size":size}
    return trades, (np.mean(notional_r) if notional_r else 0)

def summ(trades):
    if not trades: return (0,0,0,0)
    R = np.array([t["R"] for t in trades])
    return len(R), (R>0).mean()*100, R.mean(), R.sum()

# Cost scenarios (round-trip on notional)
MAKER, TAKER = 0.0002, 0.0010   # 0.02% maker (optimistic), 0.10% taker (realistic market orders)

print(f"{'TF':>5} {'stops':>9} {'gate':>4} {'n':>6} {'win%':>5} "
      f"{'gross R':>8} {'makerR':>8} {'takerR':>8} {'not/risk':>8}")
print("-"*72)
for tf, label in [("5min","5m"), ("1h","1h"), ("4h","4h")]:
    df = resample(tf)
    for (sl, tp, sname) in [(1.2, 3.6, "1.2/3.6"), (2.5, 5.0, "2.5/5.0")]:
        for gate in (False, True):
            base = P(); base.sl_mult=sl; base.tp_mult=tp; base.gate_on=gate
            base.cost=0.0
            tr, nr = backtest(df, base)
            n, wr, _, gR = summ(tr)
            # maker / taker: rerun with cost (cheap)
            base.cost=MAKER; trm,_ = backtest(df, base); _,_,_,mR = summ(trm)
            base.cost=TAKER; trt,_ = backtest(df, base); _,_,_,tR = summ(trt)
            print(f"{label:>5} {sname:>9} {'ON' if gate else 'off':>4} {n:>6} {wr:>5.1f} "
                  f"{gR:>+8.1f} {mR:>+8.1f} {tR:>+8.1f} {nr:>8.0f}x")
print("\nR = risk-multiples (total). gross=0 fees | maker=0.02% rt | taker=0.10% rt")
print("not/risk = avg position notional / dollar risked (fee leverage). Lower = fees hurt less.")
