"""
Hourly Timeframe Backtest
=========================
Resample 5m data to 1H and test simple trend strategy
"""

import pandas as pd
import numpy as np
import sys


def run_backtest():
    print("=" * 60)
    print("HOURLY TREND-FOLLOWING BACKTEST")
    print("=" * 60)
    sys.stdout.flush()
    
    # Load and resample to hourly
    print("\nLoading data...")
    df = pd.read_csv('historical_data.csv', index_col='timestamp', parse_dates=True)
    print(f"Loaded {len(df):,} 5m candles")
    
    # Resample to 1H
    df_h = df.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    print(f"Resampled to {len(df_h):,} hourly candles")
    print(f"Period: {df_h.index[0]} to {df_h.index[-1]}")
    sys.stdout.flush()
    
    # Strategy: Simple trend following
    # - EMA 20/50 for direction
    # - RSI for confirmation
    # - ATR for stops
    
    df_h['ema20'] = df_h['close'].ewm(span=20, adjust=False).mean()
    df_h['ema50'] = df_h['close'].ewm(span=50, adjust=False).mean()
    
    # RSI
    delta = df_h['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df_h['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR
    tr1 = df_h['high'] - df_h['low']
    tr2 = abs(df_h['high'] - df_h['close'].shift(1))
    tr3 = abs(df_h['low'] - df_h['close'].shift(1))
    df_h['atr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
    
    # Strategy parameters
    initial_balance = 1000.0
    balance = initial_balance
    risk_per_trade = 0.02
    leverage = 5
    sl_mult = 2.0
    tp_mult = 3.0  # 1.5:1 R:R
    
    trades = []
    position = None
    
    print(f"\nStrategy: EMA 20/50 + RSI confirmation")
    print(f"Risk: {risk_per_trade*100}% | Leverage: {leverage}x")
    print(f"SL: {sl_mult}x ATR | TP: {tp_mult}x ATR")
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
        atr = row['atr']
        rsi = row['rsi']
        
        if pd.isna(atr) or pd.isna(rsi):
            continue
        
        # Check existing position
        if position:
            # Check exit
            high = row['high']
            low = row['low']
            
            if position['side'] == 'long':
                if low <= position['sl']:
                    pnl = (position['sl'] - position['entry']) * position['size'] * leverage
                    balance += pnl
                    trades.append({'side': 'long', 'pnl': pnl, 'exit': 'sl', 'balance': balance})
                    position = None
                elif high >= position['tp']:
                    pnl = (position['tp'] - position['entry']) * position['size'] * leverage
                    balance += pnl
                    trades.append({'side': 'long', 'pnl': pnl, 'exit': 'tp', 'balance': balance})
                    position = None
            else:
                if high >= position['sl']:
                    pnl = (position['entry'] - position['sl']) * position['size'] * leverage
                    balance += pnl
                    trades.append({'side': 'short', 'pnl': pnl, 'exit': 'sl', 'balance': balance})
                    position = None
                elif low <= position['tp']:
                    pnl = (position['entry'] - position['tp']) * position['size'] * leverage
                    balance += pnl
                    trades.append({'side': 'short', 'pnl': pnl, 'exit': 'tp', 'balance': balance})
                    position = None
            continue
        
        # Entry signals
        ema_bull = row['ema20'] > row['ema50']
        ema_bear = row['ema20'] < row['ema50']
        cross_up = prev['ema20'] <= prev['ema50'] and row['ema20'] > row['ema50']
        cross_down = prev['ema20'] >= prev['ema50'] and row['ema20'] < row['ema50']
        
        # LONG: EMA cross up + RSI not overbought
        if cross_up and rsi < 70 and price > row['ema50']:
            sl = price - (atr * sl_mult)
            tp = price + (atr * tp_mult)
            risk_amt = balance * risk_per_trade
            size = risk_amt / (price - sl) if (price - sl) > 0 else 0
            
            if size > 0:
                position = {'side': 'long', 'entry': price, 'sl': sl, 'tp': tp, 'size': size}
        
        # SHORT: EMA cross down + RSI not oversold
        elif cross_down and rsi > 30 and price < row['ema50']:
            sl = price + (atr * sl_mult)
            tp = price - (atr * tp_mult)
            risk_amt = balance * risk_per_trade
            size = risk_amt / (sl - price) if (sl - price) > 0 else 0
            
            if size > 0:
                position = {'side': 'short', 'entry': price, 'sl': sl, 'tp': tp, 'size': size}
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if not trades:
        print("No trades!")
        return
    
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv('backtest_hourly_trades.csv', index=False)
    
    total = len(trades_df)
    wins = len(trades_df[trades_df['pnl'] > 0])
    win_rate = wins / total * 100
    
    tp_count = len(trades_df[trades_df['exit'] == 'tp'])
    sl_count = len(trades_df[trades_df['exit'] == 'sl'])
    
    total_pnl = trades_df['pnl'].sum()
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if (total - wins) > 0 else 0
    
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    ret = (balance - initial_balance) / initial_balance * 100
    
    # Max drawdown
    bal_series = pd.Series([initial_balance] + trades_df['balance'].tolist())
    peak = bal_series.expanding().max()
    dd = (bal_series - peak) / peak * 100
    max_dd = dd.min()
    
    print(f"\nPeriod: {df_h.index[0].strftime('%Y-%m-%d')} to {df_h.index[-1].strftime('%Y-%m-%d')}")
    print(f"Initial: ${initial_balance:,.2f}")
    print(f"Final: ${balance:,.2f}")
    print(f"Return: {ret:+.2f}%")
    
    print(f"\nTrades: {total}")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  TP: {tp_count} | SL: {sl_count}")
    
    print(f"\nP&L: ${total_pnl:+,.2f}")
    print(f"  Avg Win: ${avg_win:+,.2f}")
    print(f"  Avg Loss: ${avg_loss:+,.2f}")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Max Drawdown: {max_dd:.1f}%")
    
    print(f"\nSaved: backtest_hourly_trades.csv")


if __name__ == "__main__":
    run_backtest()
