"""
Market Profile Strategy Backtest
================================
Test Dalton auction theory concepts
"""

import pandas as pd
import numpy as np
import sys


def calculate_profile(df, price_bins=50, va_pct=0.70):
    """Calculate POC, VAH, VAL"""
    price_min = df['low'].min()
    price_max = df['high'].max()
    bins = np.linspace(price_min, price_max, price_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    volume_profile = np.zeros(price_bins)
    
    for _, row in df.iterrows():
        low_bin = np.searchsorted(bins, row['low'], side='right') - 1
        high_bin = np.searchsorted(bins, row['high'], side='left')
        
        low_bin = max(0, min(low_bin, price_bins - 1))
        high_bin = max(0, min(high_bin, price_bins - 1))
        
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
    low_idx = poc_idx
    high_idx = poc_idx
    
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
    
    val = bin_centers[low_idx]
    vah = bin_centers[high_idx]
    
    return poc, vah, val


def run_backtest():
    print("=" * 60)
    print("MARKET PROFILE (DALTON) BACKTEST")
    print("=" * 60)
    sys.stdout.flush()
    
    print("\nLoading 5m data...")
    df = pd.read_csv('historical_data.csv', index_col='timestamp', parse_dates=True)
    print(f"{len(df):,} candles")
    
    # ATR
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    df['atr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
    
    configs = [
        {'name': 'FADE to POC (4h profile)', 'period': 48, 'mode': 'fade', 'tp_poc': True, 'sl': 1.5},
        {'name': 'FADE to POC (8h profile)', 'period': 96, 'mode': 'fade', 'tp_poc': True, 'sl': 1.5},
        {'name': 'FADE ATR target', 'period': 48, 'mode': 'fade', 'tp_poc': False, 'sl': 1.5, 'tp': 2.0},
        {'name': 'BREAKOUT mode', 'period': 48, 'mode': 'breakout', 'tp_poc': False, 'sl': 1.5, 'tp': 2.0},
    ]
    
    for cfg in configs:
        print(f"\n--- {cfg['name']} ---")
        sys.stdout.flush()
        
        initial_balance = 1000.0
        balance = initial_balance
        risk_per_trade = 0.01
        leverage = 5
        period = cfg['period']
        
        trades = []
        position = None
        
        # Cache profiles to speed up
        profile_cache = {}
        
        for i in range(period + 20, len(df) - 50, 1):  # Step by 1 for more signals
            if balance < 10:
                break
            
            row = df.iloc[i]
            prev = df.iloc[i-1]
            
            price = row['close']
            atr = row['atr']
            
            if pd.isna(atr):
                continue
            
            # Check position first
            if position:
                high = row['high']
                low = row['low']
                
                if position['side'] == 'long':
                    if low <= position['sl']:
                        pnl = (position['sl'] - position['entry']) * position['size'] * leverage
                        balance += pnl
                        trades.append({'pnl': pnl, 'exit': 'sl'})
                        position = None
                    elif high >= position['tp']:
                        pnl = (position['tp'] - position['entry']) * position['size'] * leverage
                        balance += pnl
                        trades.append({'pnl': pnl, 'exit': 'tp'})
                        position = None
                else:
                    if high >= position['sl']:
                        pnl = (position['entry'] - position['sl']) * position['size'] * leverage
                        balance += pnl
                        trades.append({'pnl': pnl, 'exit': 'sl'})
                        position = None
                    elif low <= position['tp']:
                        pnl = (position['entry'] - position['tp']) * position['size'] * leverage
                        balance += pnl
                        trades.append({'pnl': pnl, 'exit': 'tp'})
                        position = None
                continue
            
            # Calculate profile (cache every 12 candles = 1 hour)
            cache_key = i // 12
            if cache_key not in profile_cache:
                profile_df = df.iloc[i-period:i]
                poc, vah, val = calculate_profile(profile_df)
                profile_cache[cache_key] = (poc, vah, val)
            else:
                poc, vah, val = profile_cache[cache_key]
            
            above_va = price > vah
            below_va = price < val
            prev_above_va = prev['close'] > vah
            prev_below_va = prev['close'] < val
            
            if cfg['mode'] == 'fade':
                # LONG: Below VA rejection
                if prev_below_va and price > prev['close'] and price > val * 0.998:
                    sl = price - (atr * cfg['sl'])
                    tp = poc if cfg['tp_poc'] else price + (atr * cfg.get('tp', 2.0))
                    
                    if tp > price:
                        risk_amt = balance * risk_per_trade
                        size = risk_amt / (price - sl) if (price - sl) > 0 else 0
                        if size > 0:
                            position = {'side': 'long', 'entry': price, 'sl': sl, 'tp': tp, 'size': size}
                
                # SHORT: Above VA rejection
                elif prev_above_va and price < prev['close'] and price < vah * 1.002:
                    sl = price + (atr * cfg['sl'])
                    tp = poc if cfg['tp_poc'] else price - (atr * cfg.get('tp', 2.0))
                    
                    if tp < price:
                        risk_amt = balance * risk_per_trade
                        size = risk_amt / (sl - price) if (sl - price) > 0 else 0
                        if size > 0:
                            position = {'side': 'short', 'entry': price, 'sl': sl, 'tp': tp, 'size': size}
            
            elif cfg['mode'] == 'breakout':
                # LONG breakout above VAH
                if above_va and prev_above_va and price > prev['close']:
                    sl = vah - (atr * 0.5)
                    tp = price + (atr * cfg.get('tp', 2.0))
                    
                    risk_amt = balance * risk_per_trade
                    size = risk_amt / (price - sl) if (price - sl) > 0 else 0
                    if size > 0:
                        position = {'side': 'long', 'entry': price, 'sl': sl, 'tp': tp, 'size': size}
                
                # SHORT breakout below VAL
                elif below_va and prev_below_va and price < prev['close']:
                    sl = val + (atr * 0.5)
                    tp = price - (atr * cfg.get('tp', 2.0))
                    
                    risk_amt = balance * risk_per_trade
                    size = risk_amt / (sl - price) if (sl - price) > 0 else 0
                    if size > 0:
                        position = {'side': 'short', 'entry': price, 'sl': sl, 'tp': tp, 'size': size}
        
        # Results
        if trades:
            trades_df = pd.DataFrame(trades)
            total = len(trades_df)
            wins = len(trades_df[trades_df['pnl'] > 0])
            win_rate = wins / total * 100
            
            gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
            pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            ret = (balance - initial_balance) / initial_balance * 100
            
            tp_count = len(trades_df[trades_df['exit'] == 'tp'])
            sl_count = len(trades_df[trades_df['exit'] == 'sl'])
            
            print(f"  {total} trades | Win: {win_rate:.1f}% | PF: {pf:.2f} | Return: {ret:+.1f}%")
            print(f"  TP: {tp_count} | SL: {sl_count} | Final: ${balance:.2f}")
        else:
            print("  No trades")
        sys.stdout.flush()
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_backtest()
