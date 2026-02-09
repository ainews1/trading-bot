"""Quick find best config"""
import pandas as pd
import numpy as np

df = pd.read_csv('historical_data.csv', usecols=['open','high','low','close'])
close = df['close'].values
high = df['high'].values
low = df['low'].values
n = len(close)

def test(ema_p, rsi_p, rsi_lo, rsi_hi, sl, tp, lev):
    ema = pd.Series(close).ewm(span=ema_p, adjust=False).mean().values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=rsi_p, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_p, adjust=False).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - 100 / (1 + rs)
    rsi_prev = np.roll(rsi, 1)
    
    warmup = ema_p + 10
    long_mask = (close > ema) & (rsi_prev >= rsi_lo) & (rsi < rsi_lo)
    short_mask = (close < ema) & (rsi_prev <= rsi_hi) & (rsi > rsi_hi)
    long_mask[:warmup] = False
    short_mask[:warmup] = False
    
    all_idx = np.concatenate([np.flatnonzero(long_mask), np.flatnonzero(short_mask)])
    all_side = np.concatenate([np.ones(np.sum(long_mask)), -np.ones(np.sum(short_mask))])
    order = np.argsort(all_idx)
    all_idx, all_side = all_idx[order], all_side[order]
    
    balance = 1000.0
    last_exit = warmup
    wins = losses = 0
    
    for k in range(len(all_idx)):
        i = all_idx[k]
        if i <= last_exit or balance <= 0: continue
        side = all_side[k]
        entry = close[i]
        sl_p = entry * (1 - sl if side == 1 else 1 + sl)
        tp_p = entry * (1 + tp if side == 1 else 1 - tp)
        pos = (balance * 0.02) / abs(entry - sl_p)
        
        for j in range(i + 1, min(i + 100, n)):
            hit_sl = (low[j] <= sl_p) if side == 1 else (high[j] >= sl_p)
            hit_tp = (high[j] >= tp_p) if side == 1 else (low[j] <= tp_p)
            if hit_sl:
                balance += (sl_p - entry) * pos * lev * side
                last_exit = j
                losses += 1
                break
            if hit_tp:
                balance += (tp_p - entry) * pos * lev * side
                last_exit = j
                wins += 1
                break
    
    return balance, wins, losses

# Test the configs around EMA 20-50, RSI 7-14, etc
print("Testing promising configs...")
best = (0, None)

for ema in [20, 50]:
    for rsi in [7, 14]:
        for lo in [20, 25, 30, 35]:
            for hi in [65, 70, 75, 80]:
                for sl in [0.01, 0.015, 0.02, 0.025]:
                    for tp in [0.04, 0.05]:
                        for lev in [3, 5]:
                            bal, w, l = test(ema, rsi, lo, hi, sl, tp, lev)
                            ret = (bal/1000 - 1) * 100
                            if ret > best[0]:
                                best = (ret, (ema, rsi, lo, hi, sl, tp, lev, bal, w, l))

ret, params = best
ema, rsi, lo, hi, sl, tp, lev, bal, w, l = params
total = w + l
print(f"\nBEST CONFIG FOUND:")
print(f"Return: {ret:,.1f}%")
print(f"${1000:,} -> ${bal:,.2f}")
print(f"Trades: {total} | Wins: {w} ({w/total*100:.1f}%)")
print(f"\nParameters:")
print(f"  EMA Period: {ema}")
print(f"  RSI Period: {rsi}")  
print(f"  RSI Oversold: {lo}")
print(f"  RSI Overbought: {hi}")
print(f"  Stop Loss: {sl*100}%")
print(f"  Take Profit: {tp*100}%")
print(f"  Leverage: {lev}x")
