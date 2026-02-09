"""Test 2024-2026 data, target 60% win rate"""
import pandas as pd
import numpy as np

# Load and filter 2024+
df = pd.read_csv('historical_data.csv', parse_dates=['timestamp'])
df = df[df['timestamp'] >= '2024-01-01']
close = df['close'].values
high = df['high'].values
low = df['low'].values
n = len(close)
print(f"Testing on {n:,} candles (2024-now)")

def test(ema_p, rsi_p, rsi_lo, rsi_hi, sl, tp, lev):
    ema = pd.Series(close).ewm(span=ema_p, adjust=False).mean().values
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    avg_gain = pd.Series(gain).ewm(span=rsi_p, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_p, adjust=False).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - 100 / (1 + rs)
    rsi_prev = np.roll(rsi, 1)
    
    warmup = max(ema_p, rsi_p) + 10
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
        i = int(all_idx[k])
        if i <= last_exit or balance <= 0: continue
        side = all_side[k]
        entry = close[i]
        sl_p = entry * (1 - sl) if side == 1 else entry * (1 + sl)
        tp_p = entry * (1 + tp) if side == 1 else entry * (1 - tp)
        risk = abs(entry - sl_p)
        if risk == 0: continue
        pos = (balance * 0.02) / risk
        
        for j in range(i + 1, min(i + 100, n)):
            if side == 1:
                if low[j] <= sl_p:
                    balance += (sl_p - entry) * pos * lev
                    last_exit, losses = j, losses + 1
                    break
                if high[j] >= tp_p:
                    balance += (tp_p - entry) * pos * lev
                    last_exit, wins = j, wins + 1
                    break
            else:
                if high[j] >= sl_p:
                    balance += (entry - sl_p) * pos * lev
                    last_exit, losses = j, losses + 1
                    break
                if low[j] <= tp_p:
                    balance += (entry - tp_p) * pos * lev
                    last_exit, wins = j, wins + 1
                    break
    
    total = wins + losses
    wr = wins / total * 100 if total > 0 else 0
    ret = (balance / 1000 - 1) * 100
    return balance, ret, wins, losses, wr

print("Finding 60%+ win rate configs...")
results = []

# Search for high win rate
for ema in [10, 20, 50, 100]:
    for rsi in [7, 14, 21]:
        for lo in [15, 20, 25, 30]:
            for hi in [70, 75, 80, 85]:
                for sl in [0.02, 0.025, 0.03]:  # Wider stops
                    for tp in [0.015, 0.02, 0.025]:  # Tighter TP for higher win rate
                        for lev in [2, 3, 5]:
                            bal, ret, w, l, wr = test(ema, rsi, lo, hi, sl, tp, lev)
                            if wr >= 55 and (w + l) >= 5:  # At least 55% and 5 trades
                                results.append({
                                    'wr': wr, 'ret': ret, 'bal': bal,
                                    'w': w, 'l': l,
                                    'ema': ema, 'rsi': rsi, 'lo': lo, 'hi': hi,
                                    'sl': sl, 'tp': tp, 'lev': lev
                                })

# Sort by win rate then return
results.sort(key=lambda x: (x['wr'], x['ret']), reverse=True)

print(f"\n{'='*60}")
print(f"TOP CONFIGS WITH 55%+ WIN RATE (2024-now)")
print(f"{'='*60}")

for i, r in enumerate(results[:10]):
    print(f"\n{i+1}. Win Rate: {r['wr']:.1f}% | Return: {r['ret']:+.1f}%")
    print(f"   Trades: {r['w']+r['l']} (W:{r['w']} L:{r['l']})")
    print(f"   EMA:{r['ema']} RSI:{r['rsi']} Lo:{r['lo']} Hi:{r['hi']} SL:{r['sl']*100}% TP:{r['tp']*100}% Lev:{r['lev']}x")

if results:
    best = results[0]
    print(f"\n{'='*60}")
    print(f"RECOMMENDED CONFIG:")
    print(f"{'='*60}")
    print(f"Win Rate: {best['wr']:.1f}%")
    print(f"Return: {best['ret']:+.1f}%")
    print(f"${1000} -> ${best['bal']:,.2f}")
else:
    print("No config found with 55%+ win rate")
