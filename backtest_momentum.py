"""
Momentum Strategy Backtest
==========================
Buy strong up moves, short strong down moves
Simple: price > X% above 20-period low = LONG
"""

import pandas as pd
import numpy as np
import sys


def run_backtest():
    print("=" * 60)
    print("MOMENTUM STRATEGY BACKTEST")
    print("=" * 60)
    sys.stdout.flush()
    
    print("\nLoading data...")
    df = pd.read_csv('historical_data.csv', index_col='timestamp', parse_dates=True)
    
    # 4H data - less noise
    df_h = df.resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    print(f"{len(df_h):,} 4h candles")
    sys.stdout.flush()
    
    # Indicators - shift to avoid lookahead
    df_h['high20'] = df_h['high'].rolling(20).max().shift(1)
    df_h['low20'] = df_h['low'].rolling(20).min().shift(1)
    df_h['mid'] = (df_h['high20'] + df_h['low20']) / 2
    
    tr1 = df_h['high'] - df_h['low']
    tr2 = abs(df_h['high'] - df_h['close'].shift(1))
    tr3 = abs(df_h['low'] - df_h['close'].shift(1))
    df_h['atr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
    
    # 50 EMA for trend
    df_h['ema50'] = df_h['close'].ewm(span=50, adjust=False).mean()
    
    # Parameters - test multiple
    configs = [
        {'breakout': 0.02, 'sl': 2.0, 'tp': 4.0},  # 2% breakout
        {'breakout': 0.03, 'sl': 2.0, 'tp': 4.0},  # 3% breakout  
        {'breakout': 0.02, 'sl': 1.5, 'tp': 3.0},  # Tighter
    ]
    
    for cfg in configs:
        print(f"\n--- Breakout: {cfg['breakout']*100}%, SL: {cfg['sl']}x, TP: {cfg['tp']}x ---")
        
        initial_balance = 1000.0
        balance = initial_balance
        risk_per_trade = 0.02
        leverage = 5
        
        trades = []
        position = None
        
        for i in range(25, len(df_h) - 1):
            if balance < 10:
                break
                
            row = df_h.iloc[i]
            prev = df_h.iloc[i-1]
            
            price = row['close']
            high = row['high']
            low = row['low']
            atr = row['atr']
            high20 = row['high20']
            low20 = row['low20']
            ema50 = row['ema50']
            
            if pd.isna(atr) or pd.isna(high20):
                continue
            
            # Check position
            if position:
                if position['side'] == 'long':
                    if low <= position['sl']:
                        pnl = (position['sl'] - position['entry']) * position['size'] * leverage
                        balance += pnl
                        trades.append({'pnl': pnl, 'exit': 'sl', 'balance': balance})
                        position = None
                    elif high >= position['tp']:
                        pnl = (position['tp'] - position['entry']) * position['size'] * leverage
                        balance += pnl
                        trades.append({'pnl': pnl, 'exit': 'tp', 'balance': balance})
                        position = None
                else:
                    if high >= position['sl']:
                        pnl = (position['entry'] - position['sl']) * position['size'] * leverage
                        balance += pnl
                        trades.append({'pnl': pnl, 'exit': 'sl', 'balance': balance})
                        position = None
                    elif low <= position['tp']:
                        pnl = (position['entry'] - position['tp']) * position['size'] * leverage
                        balance += pnl
                        trades.append({'pnl': pnl, 'exit': 'tp', 'balance': balance})
                        position = None
                continue
            
            # Entry: Breakout above high20 (high20 is already shifted)
            broke_high = price > high20  # Current close > previous 20-bar high
            broke_low = price < low20    # Current close < previous 20-bar low
            
            # LONG: Break above 20-period high + above EMA
            if broke_high and price > ema50:
                sl = price - (atr * cfg['sl'])
                tp = price + (atr * cfg['tp'])
                risk_amt = balance * risk_per_trade
                size = risk_amt / (price - sl) if (price - sl) > 0 else 0
                
                if size > 0:
                    position = {'side': 'long', 'entry': price, 'sl': sl, 'tp': tp, 'size': size}
            
            # SHORT: Break below 20-period low + below EMA
            elif broke_low and price < ema50:
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
