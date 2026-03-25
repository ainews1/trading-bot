"""
Market Profile + Trend Filter Backtest
"""

import pandas as pd
import numpy as np
import sys


def calculate_profile(df, price_bins=50, va_pct=0.70):
    price_min = df['low'].min()
    price_max = df['high'].max()
    bins = np.linspace(price_min, price_max, price_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    volume_profile = np.zeros(price_bins)
    
    for _, row in df.iterrows():
        low_bin = max(0, min(np.searchsorted(bins, row['low'], side='right') - 1, price_bins - 1))
        high_bin = max(0, min(np.searchsorted(bins, row['high'], side='left'), price_bins - 1))
        
        touched_bins = high_bin - low_bin + 1
        if touched_bins > 0:
            vol_per_bin = row['volume'] / touched_bins
            for b in range(low_bin, high_bin + 1):
                if 0 <= b < price_bins:
                    volume_profile[b] += vol_per_bin
    
    poc_idx = np.argmax(volume_profile)
    poc = bin_centers[poc_idx]
    
    total_vol = volume_profile.sum()
    target_vol = total_vol * va_pct
    
    va_vol = volume_profile[poc_idx]
    low_idx = high_idx = poc_idx
    
    while va_vol < target_vol and (low_idx > 0 or high_idx < price_bins - 1):
        low_vol = volume_profile[low_idx - 1] if low_idx > 0 else 0
        high_vol = volume_profile[high_idx + 1] if high_idx < price_bins - 1 else 0
        
        if low_vol >= high_vol and low_idx > 0:
            low_idx -= 1
            va_vol += low_vol
        elif high_idx < price_bins - 1:
            high_idx += 1
            va_vol += high_vol
        else:
            break
    
    return poc, bin_centers[high_idx], bin_centers[low_idx]


def run_backtest():
    print("=" * 60)
    print("MARKET PROFILE + TREND FILTER")
    print("=" * 60)
    
    df = pd.read_csv('historical_data.csv', index_col='timestamp', parse_dates=True)
    print(f"{len(df):,} candles")
    
    # ATR
    tr = pd.concat([df['high'] - df['low'], 
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # EMAs for trend
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema100'] = df['close'].ewm(span=100, adjust=False).mean()
    
    configs = [
        {'name': 'Profile FADE + EMA50 trend', 'period': 48, 'ema': 50, 'sl': 1.5, 'tp': 2.0},
        {'name': 'Profile FADE + EMA100 trend', 'period': 48, 'ema': 100, 'sl': 1.5, 'tp': 2.0},
        {'name': 'Profile FADE + EMA50 (wider SL)', 'period': 48, 'ema': 50, 'sl': 2.0, 'tp': 2.5},
        {'name': '8h Profile + EMA50', 'period': 96, 'ema': 50, 'sl': 1.5, 'tp': 2.0},
    ]
    
    for cfg in configs:
        print(f"\n--- {cfg['name']} ---")
        sys.stdout.flush()
        
        balance = 1000.0
        trades = []
        position = None
        period = cfg['period']
        ema_col = f"ema{cfg['ema']}"
        
        profile_cache = {}
        
        for i in range(period + 20, len(df) - 50):
            if balance < 10:
                break
            
            row = df.iloc[i]
            prev = df.iloc[i-1]
            
            price = row['close']
            atr = row['atr']
            ema = row[ema_col]
            
            if pd.isna(atr) or pd.isna(ema):
                continue
            
            # Check position
            if position:
                if position['side'] == 'long':
                    if row['low'] <= position['sl']:
                        pnl = (position['sl'] - position['entry']) * position['size'] * 5
                        balance += pnl
                        trades.append({'pnl': pnl, 'exit': 'sl'})
                        position = None
                    elif row['high'] >= position['tp']:
                        pnl = (position['tp'] - position['entry']) * position['size'] * 5
                        balance += pnl
                        trades.append({'pnl': pnl, 'exit': 'tp'})
                        position = None
                else:
                    if row['high'] >= position['sl']:
                        pnl = (position['entry'] - position['sl']) * position['size'] * 5
                        balance += pnl
                        trades.append({'pnl': pnl, 'exit': 'sl'})
                        position = None
                    elif row['low'] <= position['tp']:
                        pnl = (position['entry'] - position['tp']) * position['size'] * 5
                        balance += pnl
                        trades.append({'pnl': pnl, 'exit': 'tp'})
                        position = None
                continue
            
            # Get profile
            cache_key = i // 12
            if cache_key not in profile_cache:
                poc, vah, val = calculate_profile(df.iloc[i-period:i])
                profile_cache[cache_key] = (poc, vah, val)
            else:
                poc, vah, val = profile_cache[cache_key]
            
            prev_below_va = prev['close'] < val
            prev_above_va = prev['close'] > vah
            
            # LONG: Below VA + bouncing + above EMA (uptrend)
            if prev_below_va and price > prev['close'] and price > val * 0.998 and price > ema:
                sl = price - (atr * cfg['sl'])
                tp = price + (atr * cfg['tp'])
                size = (balance * 0.01) / (price - sl) if (price - sl) > 0 else 0
                
                if size > 0 and tp > price:
                    position = {'side': 'long', 'entry': price, 'sl': sl, 'tp': tp, 'size': size}
            
            # SHORT: Above VA + falling + below EMA (downtrend)
            elif prev_above_va and price < prev['close'] and price < vah * 1.002 and price < ema:
                sl = price + (atr * cfg['sl'])
                tp = price - (atr * cfg['tp'])
                size = (balance * 0.01) / (sl - price) if (sl - price) > 0 else 0
                
                if size > 0 and tp < price:
                    position = {'side': 'short', 'entry': price, 'sl': sl, 'tp': tp, 'size': size}
        
        if trades:
            trades_df = pd.DataFrame(trades)
            total = len(trades_df)
            wins = len(trades_df[trades_df['pnl'] > 0])
            win_rate = wins / total * 100
            
            gp = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            gl = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
            pf = gp / gl if gl > 0 else float('inf')
            
            ret = (balance - 1000) / 1000 * 100
            
            print(f"  {total} trades | Win: {win_rate:.1f}% | PF: {pf:.2f} | Return: {ret:+.1f}%")
            print(f"  Final: ${balance:.2f}")
        else:
            print("  No trades")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_backtest()
