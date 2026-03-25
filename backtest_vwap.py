"""
VWAP Mean Reversion Backtest
============================
Buy below VWAP, sell above VWAP
"""

import pandas as pd
import numpy as np
import sys


def run_backtest():
    print("=" * 60)
    print("VWAP MEAN REVERSION BACKTEST")
    print("=" * 60)
    sys.stdout.flush()
    
    print("\nLoading data...")
    df = pd.read_csv('historical_data.csv', index_col='timestamp', parse_dates=True)
    print(f"{len(df):,} candles")
    sys.stdout.flush()
    
    # Calculate VWAP (rolling 50-period)
    df['typical'] = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (df['typical'] * df['volume']).rolling(50).sum() / df['volume'].rolling(50).sum()
    
    # Bollinger Bands on VWAP distance
    df['vwap_dist'] = (df['close'] - df['vwap']) / df['vwap'] * 100  # % from VWAP
    df['vwap_std'] = df['vwap_dist'].rolling(20).std()
    
    # ATR
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    df['atr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
    
    # EMA trend filter
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    configs = [
        {'dist': 0.5, 'sl': 1.0, 'tp': 0.75},  # Mean revert at 0.5% from VWAP
        {'dist': 0.3, 'sl': 0.5, 'tp': 0.5},   # Tighter
        {'dist': 0.7, 'sl': 1.5, 'tp': 1.0},   # Wider
        {'dist': 1.0, 'sl': 1.5, 'tp': 1.5},   # Even wider
    ]
    
    for cfg in configs:
        print(f"\n--- Distance: {cfg['dist']}% SL: {cfg['sl']}x TP: {cfg['tp']}x ---")
        sys.stdout.flush()
        
        initial_balance = 1000.0
        balance = initial_balance
        risk_per_trade = 0.01
        leverage = 5
        
        trades = []
        position = None
        
        for i in range(250, len(df) - 50):
            if balance < 10:
                break
            
            row = df.iloc[i]
            prev = df.iloc[i-1]
            
            price = row['close']
            vwap = row['vwap']
            vwap_dist = row['vwap_dist']
            atr = row['atr']
            ema200 = row['ema200']
            
            if pd.isna(vwap) or pd.isna(atr):
                continue
            
            # Check position
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
            
            # LONG: Price far below VWAP, in uptrend
            if vwap_dist < -cfg['dist'] and prev['vwap_dist'] < -cfg['dist'] and price > ema200:
                # Bouncing back toward VWAP
                if row['close'] > prev['close']:
                    sl = price - (atr * cfg['sl'])
                    tp = min(vwap, price + (atr * cfg['tp']))  # Target VWAP or TP
                    risk_amt = balance * risk_per_trade
                    size = risk_amt / (price - sl) if (price - sl) > 0 else 0
                    
                    if size > 0 and tp > price:
                        position = {'side': 'long', 'entry': price, 'sl': sl, 'tp': tp, 'size': size}
            
            # SHORT: Price far above VWAP, in downtrend
            elif vwap_dist > cfg['dist'] and prev['vwap_dist'] > cfg['dist'] and price < ema200:
                # Falling back toward VWAP
                if row['close'] < prev['close']:
                    sl = price + (atr * cfg['sl'])
                    tp = max(vwap, price - (atr * cfg['tp']))
                    risk_amt = balance * risk_per_trade
                    size = risk_amt / (sl - price) if (sl - price) > 0 else 0
                    
                    if size > 0 and tp < price:
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
            
            tp = len(trades_df[trades_df['exit'] == 'tp'])
            sl = len(trades_df[trades_df['exit'] == 'sl'])
            
            print(f"  {total} trades | Win: {win_rate:.1f}% | PF: {pf:.2f} | Return: {ret:+.1f}%")
            print(f"  TP: {tp} | SL: {sl} | Final: ${balance:.2f}")
        else:
            print("  No trades")
        sys.stdout.flush()
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_backtest()
