"""
Hourly Backtest V3 - Exit on Opposite Signal
=============================================
Let winners run, exit on trend reversal
"""

import pandas as pd
import numpy as np
import sys


def run_backtest():
    print("=" * 60)
    print("HOURLY TREND V3 - EXIT ON REVERSAL")
    print("=" * 60)
    sys.stdout.flush()
    
    print("\nLoading data...")
    df = pd.read_csv('historical_data.csv', index_col='timestamp', parse_dates=True)
    
    df_h = df.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    print(f"{len(df_h):,} hourly candles")
    sys.stdout.flush()
    
    # Indicators
    df_h['ema20'] = df_h['close'].ewm(span=20, adjust=False).mean()
    df_h['ema50'] = df_h['close'].ewm(span=50, adjust=False).mean()
    
    delta = df_h['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df_h['rsi'] = 100 - (100 / (1 + rs))
    
    tr1 = df_h['high'] - df_h['low']
    tr2 = abs(df_h['high'] - df_h['close'].shift(1))
    tr3 = abs(df_h['low'] - df_h['close'].shift(1))
    df_h['atr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
    
    # Parameters
    initial_balance = 1000.0
    balance = initial_balance
    risk_per_trade = 0.02
    leverage = 5
    sl_mult = 2.5  # Wider stop
    
    trades = []
    position = None
    
    print(f"\nStrategy: EMA cross + hold until reversal")
    print(f"SL: {sl_mult}x ATR | Exit: opposite cross or SL")
    print("\nScanning...")
    sys.stdout.flush()
    
    for i in range(60, len(df_h) - 1):
        if i % 2000 == 0:
            pct = int(i / len(df_h) * 100)
            print(f"  {pct}% - {len(trades)} trades, ${balance:.2f}")
            sys.stdout.flush()
        
        if balance < 10:
            break
            
        row = df_h.iloc[i]
        prev = df_h.iloc[i-1]
        
        price = row['close']
        high = row['high']
        low = row['low']
        atr = row['atr']
        rsi = row['rsi']
        
        if pd.isna(atr) or pd.isna(rsi):
            continue
        
        cross_up = prev['ema20'] <= prev['ema50'] and row['ema20'] > row['ema50']
        cross_down = prev['ema20'] >= prev['ema50'] and row['ema20'] < row['ema50']
        
        # Manage position
        if position:
            exit_signal = False
            exit_price = None
            exit_type = None
            
            if position['side'] == 'long':
                # Check SL
                if low <= position['sl']:
                    exit_signal = True
                    exit_price = position['sl']
                    exit_type = 'sl'
                # Check reversal
                elif cross_down:
                    exit_signal = True
                    exit_price = price
                    exit_type = 'reversal'
            else:
                if high >= position['sl']:
                    exit_signal = True
                    exit_price = position['sl']
                    exit_type = 'sl'
                elif cross_up:
                    exit_signal = True
                    exit_price = price
                    exit_type = 'reversal'
            
            if exit_signal:
                if position['side'] == 'long':
                    pnl = (exit_price - position['entry']) * position['size'] * leverage
                else:
                    pnl = (position['entry'] - exit_price) * position['size'] * leverage
                
                balance += pnl
                trades.append({
                    'side': position['side'], 
                    'pnl': pnl, 
                    'exit': exit_type, 
                    'balance': balance,
                    'entry': position['entry'],
                    'exit_price': exit_price
                })
                position = None
                
                # Don't immediately re-enter
                continue
        
        # Entry (only if no position)
        if not position:
            if cross_up and rsi < 70 and rsi > 35:
                sl = price - (atr * sl_mult)
                risk_amt = balance * risk_per_trade
                size = risk_amt / (price - sl) if (price - sl) > 0 else 0
                
                if size > 0:
                    position = {'side': 'long', 'entry': price, 'sl': sl, 'size': size}
            
            elif cross_down and rsi > 30 and rsi < 65:
                sl = price + (atr * sl_mult)
                risk_amt = balance * risk_per_trade
                size = risk_amt / (sl - price) if (sl - price) > 0 else 0
                
                if size > 0:
                    position = {'side': 'short', 'entry': price, 'sl': sl, 'size': size}
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if not trades:
        print("No trades!")
        return
    
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv('backtest_hourly_v3_trades.csv', index=False)
    
    total = len(trades_df)
    wins = len(trades_df[trades_df['pnl'] > 0])
    win_rate = wins / total * 100
    
    rev_count = len(trades_df[trades_df['exit'] == 'reversal'])
    sl_count = len(trades_df[trades_df['exit'] == 'sl'])
    
    total_pnl = trades_df['pnl'].sum()
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if (total - wins) > 0 else 0
    
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    ret = (balance - initial_balance) / initial_balance * 100
    
    bal_series = pd.Series([initial_balance] + trades_df['balance'].tolist())
    peak = bal_series.expanding().max()
    dd = (bal_series - peak) / peak * 100
    max_dd = dd.min()
    
    print(f"\nInitial: ${initial_balance:,.2f}")
    print(f"Final: ${balance:,.2f}")
    print(f"Return: {ret:+.2f}%")
    
    print(f"\nTrades: {total}")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Reversal exits: {rev_count} | SL exits: {sl_count}")
    
    print(f"\nAvg Win: ${avg_win:+,.2f}")
    print(f"Avg Loss: ${avg_loss:+,.2f}")
    print(f"Profit Factor: {pf:.2f}")
    print(f"Max Drawdown: {max_dd:.1f}%")


if __name__ == "__main__":
    run_backtest()
