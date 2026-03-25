"""
5-Minute Scalping Backtest
==========================
Test multiple parameter combinations
"""

import pandas as pd
import numpy as np
import sys


def run_backtest():
    print("=" * 60)
    print("5-MINUTE SCALPING BACKTEST")
    print("=" * 60)
    sys.stdout.flush()
    
    print("\nLoading 5m data...")
    df = pd.read_csv('historical_data.csv', index_col='timestamp', parse_dates=True)
    print(f"{len(df):,} candles")
    sys.stdout.flush()
    
    # Test configurations
    configs = [
        {'rsi': 7, 'os': 25, 'ob': 75, 'sl': 1.0, 'tp': 1.5},
        {'rsi': 7, 'os': 20, 'ob': 80, 'sl': 1.0, 'tp': 2.0},
        {'rsi': 14, 'os': 30, 'ob': 70, 'sl': 1.5, 'tp': 2.0},
        {'rsi': 7, 'os': 20, 'ob': 80, 'sl': 0.5, 'tp': 1.0},  # Tight scalp
    ]
    
    for cfg in configs:
        print(f"\n--- RSI({cfg['rsi']}) OS:{cfg['os']} OB:{cfg['ob']} SL:{cfg['sl']}x TP:{cfg['tp']}x ---")
        sys.stdout.flush()
        
        # Calculate indicators
        df_test = df.copy()
        
        # RSI
        delta = df_test['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(cfg['rsi']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(cfg['rsi']).mean()
        rs = gain / loss
        df_test['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR
        tr1 = df_test['high'] - df_test['low']
        tr2 = abs(df_test['high'] - df_test['close'].shift(1))
        tr3 = abs(df_test['low'] - df_test['close'].shift(1))
        df_test['atr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
        
        # EMA
        df_test['ema'] = df_test['close'].ewm(span=50, adjust=False).mean()
        
        initial_balance = 1000.0
        balance = initial_balance
        risk_per_trade = 0.01
        leverage = 5
        
        trades = []
        position = None
        
        for i in range(60, len(df_test) - 50):
            if balance < 10:
                break
            
            row = df_test.iloc[i]
            prev = df_test.iloc[i-1]
            prev2 = df_test.iloc[i-2]
            
            price = row['close']
            rsi = row['rsi']
            atr = row['atr']
            
            if pd.isna(rsi) or pd.isna(atr):
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
            
            # Entry signals
            rsi_oversold = prev['rsi'] < cfg['os']
            rsi_turning_up = row['rsi'] > prev['rsi'] and prev['rsi'] < prev2['rsi']
            
            rsi_overbought = prev['rsi'] > cfg['ob']
            rsi_turning_down = row['rsi'] < prev['rsi'] and prev['rsi'] > prev2['rsi']
            
            # LONG
            if rsi_oversold and rsi_turning_up:
                sl = price - (atr * cfg['sl'])
                tp = price + (atr * cfg['tp'])
                risk_amt = balance * risk_per_trade
                size = risk_amt / (price - sl) if (price - sl) > 0 else 0
                
                if size > 0:
                    position = {'side': 'long', 'entry': price, 'sl': sl, 'tp': tp, 'size': size}
            
            # SHORT
            elif rsi_overbought and rsi_turning_down:
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
