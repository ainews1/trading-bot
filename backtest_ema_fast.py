"""
EMA 8/24 Fast Scalping Backtest
"""

import pandas as pd
import numpy as np
import sys


def run_backtest():
    print("=" * 60)
    print("EMA 8/24 AGGRESSIVE SCALPING BACKTEST")
    print("=" * 60)
    sys.stdout.flush()
    
    print("\nLoading 5m data...")
    df = pd.read_csv('historical_data.csv', index_col='timestamp', parse_dates=True)
    print(f"{len(df):,} candles")
    sys.stdout.flush()
    
    # Calculate EMAs
    df['ema8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema24'] = df['close'].ewm(span=24, adjust=False).mean()
    
    # ATR
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    df['atr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
    
    configs = [
        {'sl': 1.0, 'tp': 1.5},
        {'sl': 1.0, 'tp': 2.0},
        {'sl': 0.5, 'tp': 1.0},
        {'sl': 1.5, 'tp': 2.0},
        {'sl': 0.75, 'tp': 1.5},
    ]
    
    for cfg in configs:
        print(f"\n--- SL: {cfg['sl']}x ATR | TP: {cfg['tp']}x ATR ---")
        sys.stdout.flush()
        
        initial_balance = 1000.0
        balance = initial_balance
        risk_per_trade = 0.01
        leverage = 5
        
        trades = []
        position = None
        
        for i in range(50, len(df) - 50):
            if balance < 10:
                break
            
            row = df.iloc[i]
            prev = df.iloc[i-1]
            
            price = row['close']
            atr = row['atr']
            
            if pd.isna(atr):
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
            
            # Check for crossover
            cross_up = prev['ema8'] <= prev['ema24'] and row['ema8'] > row['ema24']
            cross_down = prev['ema8'] >= prev['ema24'] and row['ema8'] < row['ema24']
            
            # LONG
            if cross_up:
                sl = price - (atr * cfg['sl'])
                tp = price + (atr * cfg['tp'])
                risk_amt = balance * risk_per_trade
                size = risk_amt / (price - sl) if (price - sl) > 0 else 0
                
                if size > 0:
                    position = {'side': 'long', 'entry': price, 'sl': sl, 'tp': tp, 'size': size}
            
            # SHORT
            elif cross_down:
                sl = price + (atr * cfg['sl'])
                tp = price - (atr * cfg['tp'])
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
