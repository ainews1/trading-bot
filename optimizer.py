"""
Strategy Optimizer - Find Profitable Parameters
================================================
Tests many combinations to find +60% return
"""

import pandas as pd
import numpy as np
import time
from itertools import product

# Load data once
print("Loading data...")
df = pd.read_csv('historical_data.csv', usecols=['open','high','low','close'])
close = df['close'].values.astype(np.float64)
high = df['high'].values.astype(np.float64)
low = df['low'].values.astype(np.float64)
n = len(close)
print(f"Loaded {n:,} candles")


def backtest(ema_period, rsi_period, rsi_lo, rsi_hi, sl_pct, tp_pct, leverage):
    """Run single backtest, return final balance"""
    
    # EMA
    ema = pd.Series(close).ewm(span=ema_period, adjust=False).mean().values
    
    # RSI
    delta = np.empty(n)
    delta[0] = 0
    delta[1:] = close[1:] - close[:-1]
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, adjust=False).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - 100 / (1 + rs)
    
    rsi_prev = np.empty(n)
    rsi_prev[0] = 50
    rsi_prev[1:] = rsi[:-1]
    
    warmup = ema_period + 10
    
    # Signals
    long_mask = (close > ema) & (rsi_prev >= rsi_lo) & (rsi < rsi_lo)
    short_mask = (close < ema) & (rsi_prev <= rsi_hi) & (rsi > rsi_hi)
    long_mask[:warmup] = False
    short_mask[:warmup] = False
    
    long_idx = np.flatnonzero(long_mask)
    short_idx = np.flatnonzero(short_mask)
    
    # Merge signals
    all_idx = np.concatenate([long_idx, short_idx])
    all_side = np.concatenate([np.ones(len(long_idx), dtype=np.int8), 
                               -np.ones(len(short_idx), dtype=np.int8)])
    order = np.argsort(all_idx)
    all_idx = all_idx[order]
    all_side = all_side[order]
    
    # Simulate
    balance = 1000.0
    last_exit = warmup
    wins = losses = 0
    
    for k in range(len(all_idx)):
        i = all_idx[k]
        if i <= last_exit or balance <= 0:
            continue
        
        side = all_side[k]
        entry = close[i]
        
        if side == 1:
            sl_p = entry * (1 - sl_pct)
            tp_p = entry * (1 + tp_pct)
        else:
            sl_p = entry * (1 + sl_pct)
            tp_p = entry * (1 - tp_pct)
        
        risk = abs(entry - sl_p)
        if risk == 0:
            continue
        pos = (balance * 0.02) / risk
        
        end = min(i + 100, n)
        exit_p = exit_i = 0
        
        for j in range(i + 1, end):
            if side == 1:
                if low[j] <= sl_p:
                    exit_p, exit_i = sl_p, j
                    break
                if high[j] >= tp_p:
                    exit_p, exit_i = tp_p, j
                    wins += 1
                    break
            else:
                if high[j] >= sl_p:
                    exit_p, exit_i = sl_p, j
                    break
                if low[j] <= tp_p:
                    exit_p, exit_i = tp_p, j
                    wins += 1
                    break
        
        if exit_i:
            pnl = (exit_p - entry) * pos * leverage * side
            balance += pnl
            last_exit = exit_i
            if exit_p == sl_p:
                losses += 1
    
    total = wins + losses
    win_rate = wins / total * 100 if total > 0 else 0
    ret = (balance / 1000 - 1) * 100
    
    return balance, ret, total, win_rate


# Parameter grid
print("\nOptimizing...")
t0 = time.perf_counter()

best = {'ret': -999, 'params': None}
results = []

# Test different parameters
ema_periods = [20, 50, 100, 200]
rsi_periods = [7, 14, 21]
rsi_los = [20, 25, 30, 35]
rsi_his = [65, 70, 75, 80]
sl_pcts = [0.01, 0.015, 0.02, 0.025]
tp_pcts = [0.02, 0.03, 0.04, 0.05]
leverages = [2, 3, 5]

total_combos = len(ema_periods) * len(rsi_periods) * len(rsi_los) * len(rsi_his) * len(sl_pcts) * len(tp_pcts) * len(leverages)
print(f"Testing {total_combos} combinations...")

tested = 0
for ema_p in ema_periods:
    for rsi_p in rsi_periods:
        for rsi_lo in rsi_los:
            for rsi_hi in rsi_his:
                for sl in sl_pcts:
                    for tp in tp_pcts:
                        for lev in leverages:
                            bal, ret, trades, wr = backtest(ema_p, rsi_p, rsi_lo, rsi_hi, sl, tp, lev)
                            
                            if ret > best['ret']:
                                best['ret'] = ret
                                best['params'] = (ema_p, rsi_p, rsi_lo, rsi_hi, sl, tp, lev)
                                best['balance'] = bal
                                best['trades'] = trades
                                best['winrate'] = wr
                            
                            if ret > 0:
                                results.append({
                                    'ema': ema_p, 'rsi': rsi_p, 
                                    'rsi_lo': rsi_lo, 'rsi_hi': rsi_hi,
                                    'sl': sl, 'tp': tp, 'lev': lev,
                                    'return': ret, 'trades': trades, 'wr': wr
                                })
                            
                            tested += 1
                            if tested % 500 == 0:
                                print(f"  {tested}/{total_combos} - Best: {best['ret']:.1f}%")

print(f"\nDone in {time.perf_counter()-t0:.1f}s")
print(f"\n{'='*60}")
print("BEST RESULT:")
print(f"{'='*60}")

if best['params']:
    ema_p, rsi_p, rsi_lo, rsi_hi, sl, tp, lev = best['params']
    print(f"Return: {best['ret']:+.1f}%")
    print(f"Balance: ${best['balance']:,.2f}")
    print(f"Trades: {best['trades']} | Win Rate: {best['winrate']:.1f}%")
    print(f"\nParameters:")
    print(f"  EMA: {ema_p}")
    print(f"  RSI Period: {rsi_p}")
    print(f"  RSI Oversold: {rsi_lo}")
    print(f"  RSI Overbought: {rsi_hi}")
    print(f"  Stop Loss: {sl*100}%")
    print(f"  Take Profit: {tp*100}%")
    print(f"  Leverage: {lev}x")

# Show top 10 profitable
if results:
    print(f"\n{'='*60}")
    print(f"TOP 10 PROFITABLE CONFIGS:")
    print(f"{'='*60}")
    results.sort(key=lambda x: x['return'], reverse=True)
    for i, r in enumerate(results[:10]):
        print(f"{i+1}. {r['return']:+.1f}% | EMA:{r['ema']} RSI:{r['rsi']} Lo:{r['rsi_lo']} Hi:{r['rsi_hi']} SL:{r['sl']*100}% TP:{r['tp']*100}% Lev:{r['lev']}x")
else:
    print("\nNo profitable configuration found with EMA+RSI strategy.")
    print("Strategy may not be suitable for BTC - need different approach.")
