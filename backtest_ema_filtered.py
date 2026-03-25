"""
EMA 8/24 with Momentum Filter Backtest
"""

import pandas as pd
import numpy as np
import sys


def run_backtest():
    print("=" * 60)
    print("EMA 8/24 + FILTERS BACKTEST")
    print("=" * 60)
    sys.stdout.flush()
    
    print("\nLoading data...")
    df = pd.read_csv('historical_data.csv', index_col='timestamp', parse_dates=True)
    print(f"{len(df):,} candles")
    
    # Indicators
    df['ema8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema24'] = df['close'].ewm(span=24, adjust=False).mean()
    df['ema100'] = df['close'].ewm(span=100, adjust=False).mean()  # Trend filter
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(7).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(7).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    df['atr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
    
    # Momentum
    df['mom'] = df['close'].pct_change(5) * 100
    
    configs = [
        {'name': 'EMA only', 'trend': False, 'rsi': False, 'mom': False, 'sl': 1.0, 'tp': 1.5},
        {'name': '+ Trend (EMA100)', 'trend': True, 'rsi': False, 'mom': False, 'sl': 1.0, 'tp': 1.5},
        {'name': '+ RSI filter', 'trend': True, 'rsi': True, 'mom': False, 'sl': 1.0, 'tp': 1.5},
        {'name': '+ Momentum', 'trend': True, 'rsi': False, 'mom': True, 'sl': 1.0, 'tp': 1.5},
        {'name': 'All filters', 'trend': True, 'rsi': True, 'mom': True, 'sl': 1.5, 'tp': 2.0},
    ]
    
    for cfg in configs:
        print(f"\n--- {cfg['name']} ---")
        sys.stdout.flush()
        
        initial_balance = 1000.0
        balance = initial_balance
        risk_per_trade = 0.01
        leverage = 5
        
        trades = []
        position = None
        
        for i in range(100, len(df) - 50):
            if balance < 10:
                break
            
            row = df.iloc[i]
            prev = df.iloc[i-1]
            
            price = row['close']
            atr = row['atr']
            rsi = row['rsi']
            mom = row['mom']
            ema100 = row['ema100']
            
            if pd.isna(atr) or pd.isna(rsi):
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
            
            # Crossover
            cross_up = prev['ema8'] <= prev['ema24'] and row['ema8'] > row['ema24']
            cross_down = prev['ema8'] >= prev['ema24'] and row['ema8'] < row['ema24']
            
            # Filters
            trend_ok_long = not cfg['trend'] or price > ema100
            trend_ok_short = not cfg['trend'] or price < ema100
            rsi_ok_long = not cfg['rsi'] or rsi < 65
            rsi_ok_short = not cfg['rsi'] or rsi > 35
            mom_ok_long = not cfg['mom'] or mom > 0
            mom_ok_short = not cfg['mom'] or mom < 0
            
            # LONG
            if cross_up and trend_ok_long and rsi_ok_long and mom_ok_long:
                sl = price - (atr * cfg['sl'])
                tp = price + (atr * cfg['tp'])
                risk_amt = balance * risk_per_trade
                size = risk_amt / (price - sl) if (price - sl) > 0 else 0
                
                if size > 0:
                    position = {'side': 'long', 'entry': price, 'sl': sl, 'tp': tp, 'size': size}
            
            # SHORT
            elif cross_down and trend_ok_short and rsi_ok_short and mom_ok_short:
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
            
            print(f"  {total} trades | Win: {win_rate:.1f}% | PF: {pf:.2f} | Return: {ret:+.1f}%")
            print(f"  Final: ${balance:.2f}")
        else:
            print("  No trades")
        sys.stdout.flush()
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_backtest()
