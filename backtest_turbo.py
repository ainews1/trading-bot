"""
Turbo Backtest - Maximum Speed
==============================
Pandas load + numpy compute
"""

import pandas as pd
import numpy as np
import time

# Config
EMA = 200
RSI_P = 14
RSI_LO = 30
RSI_HI = 70
SL = 0.015
TP = 0.03
LEV = 5
BAL = 1000.0


def main():
    t0 = time.perf_counter()
    
    # Fast load with pandas
    df = pd.read_csv('historical_data.csv', usecols=['open','high','low','close'])
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    n = len(close)
    print(f"Load: {n:,} in {time.perf_counter()-t0:.3f}s")
    
    # EMA vectorized with pandas (fast C implementation)
    t1 = time.perf_counter()
    ema = pd.Series(close).ewm(span=EMA, adjust=False).mean().values
    
    # RSI vectorized
    delta = np.empty(n)
    delta[0] = 0
    delta[1:] = close[1:] - close[:-1]
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=RSI_P, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=RSI_P, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - 100 / (1 + rs)
    
    print(f"Indicators: {time.perf_counter()-t1:.3f}s")
    
    # Signals - pure numpy
    t1 = time.perf_counter()
    rsi_prev = np.empty(n)
    rsi_prev[0] = 50
    rsi_prev[1:] = rsi[:-1]
    
    warmup = EMA + 10
    
    long_mask = (close > ema) & (rsi_prev >= RSI_LO) & (rsi < RSI_LO)
    short_mask = (close < ema) & (rsi_prev <= RSI_HI) & (rsi > RSI_HI)
    
    long_mask[:warmup] = False
    short_mask[:warmup] = False
    
    long_idx = np.flatnonzero(long_mask)
    short_idx = np.flatnonzero(short_mask)
    
    print(f"Signals: {len(long_idx)}L {len(short_idx)}S in {time.perf_counter()-t1:.3f}s")
    
    # Merge & sort signals
    t1 = time.perf_counter()
    all_idx = np.concatenate([long_idx, short_idx])
    all_side = np.concatenate([np.ones(len(long_idx), dtype=np.int8), 
                               -np.ones(len(short_idx), dtype=np.int8)])
    order = np.argsort(all_idx, kind='quicksort')
    all_idx = all_idx[order]
    all_side = all_side[order]
    
    # Trade simulation
    balance = BAL
    last_exit = warmup
    wins = losses = 0
    gross_p = gross_l = 0.0
    
    for k in range(len(all_idx)):
        i = all_idx[k]
        if i <= last_exit or balance <= 0:
            continue
        
        side = all_side[k]
        entry = close[i]
        
        if side == 1:
            sl_p = entry * (1 - SL)
            tp_p = entry * (1 + TP)
        else:
            sl_p = entry * (1 + SL)
            tp_p = entry * (1 - TP)
        
        pos = (balance * 0.02) / abs(entry - sl_p)
        
        # Scan for exit
        end = min(i + 50, n)
        exit_p = exit_i = 0
        win = False
        
        for j in range(i + 1, end):
            if side == 1:
                if low[j] <= sl_p:
                    exit_p, exit_i = sl_p, j
                    break
                if high[j] >= tp_p:
                    exit_p, exit_i, win = tp_p, j, True
                    break
            else:
                if high[j] >= sl_p:
                    exit_p, exit_i = sl_p, j
                    break
                if low[j] <= tp_p:
                    exit_p, exit_i, win = tp_p, j, True
                    break
        
        if exit_i:
            pnl = (exit_p - entry) * pos * LEV * side
            balance += pnl
            last_exit = exit_i
            if win:
                wins += 1
                gross_p += pnl
            else:
                losses += 1
                gross_l -= pnl
    
    print(f"Trades: {time.perf_counter()-t1:.3f}s")
    
    total = wins + losses
    pf = gross_p / gross_l if gross_l > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"TOTAL: {time.perf_counter()-t0:.3f}s")
    print(f"{'='*50}")
    print(f"${BAL:,.0f} -> ${balance:,.2f} ({(balance/BAL-1)*100:+.1f}%)")
    print(f"Trades: {total} | W: {wins} ({wins/total*100 if total else 0:.1f}%) | L: {losses}")
    print(f"PF: {pf:.2f}")


if __name__ == "__main__":
    main()
