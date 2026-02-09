"""
Fast Backtest - Vectorized Operations
=====================================
Much faster than iterating through each candle
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Config values
EMA_PERIOD = 200
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
STOP_LOSS_PCT = 0.015  # 1.5%
TAKE_PROFIT_PCT = 0.03  # 3%
RISK_PER_TRADE = 0.02  # 2%
LEVERAGE = 5


def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def run_backtest():
    print("Loading data...")
    df = pd.read_csv('historical_data.csv', index_col='timestamp', parse_dates=True)
    print(f"Loaded {len(df)} candles")
    print(f"Period: {df.index[0]} to {df.index[-1]}")
    
    # Calculate indicators
    print("Calculating indicators...")
    df['ema'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
    df['rsi_prev'] = df['rsi'].shift(1)
    
    # Generate signals
    print("Finding signals...")
    
    # Long: Price > EMA AND RSI crosses below 30 (prev >= 30, current < 30)
    df['long_signal'] = (
        (df['close'] > df['ema']) & 
        (df['rsi_prev'] >= RSI_OVERSOLD) & 
        (df['rsi'] < RSI_OVERSOLD)
    )
    
    # Short: Price < EMA AND RSI crosses above 70 (prev <= 70, current > 70)
    df['short_signal'] = (
        (df['close'] < df['ema']) & 
        (df['rsi_prev'] <= RSI_OVERBOUGHT) & 
        (df['rsi'] > RSI_OVERBOUGHT)
    )
    
    # Simulate trades
    print("Simulating trades...")
    
    initial_balance = 1000.0
    balance = initial_balance
    trades = []
    
    # Skip warmup period
    warmup = max(EMA_PERIOD, RSI_PERIOD) + 10
    df_trade = df.iloc[warmup:].copy()
    
    i = 0
    while i < len(df_trade) - 50:  # Leave room for exit check
        row = df_trade.iloc[i]
        idx = df_trade.index[i]
        
        signal = None
        if row['long_signal']:
            signal = 'long'
        elif row['short_signal']:
            signal = 'short'
        
        if signal:
            entry_price = row['close']
            
            if signal == 'long':
                stop_loss = entry_price * (1 - STOP_LOSS_PCT)
                take_profit = entry_price * (1 + TAKE_PROFIT_PCT)
            else:
                stop_loss = entry_price * (1 + STOP_LOSS_PCT)
                take_profit = entry_price * (1 - TAKE_PROFIT_PCT)
            
            # Calculate position size
            risk_amount = balance * RISK_PER_TRADE
            price_risk = abs(entry_price - stop_loss)
            position_size = risk_amount / price_risk if price_risk > 0 else 0
            
            # Look for exit in next 50 candles
            exit_type = None
            exit_price = None
            exit_idx = None
            
            for j in range(i + 1, min(i + 50, len(df_trade))):
                candle = df_trade.iloc[j]
                high = candle['high']
                low = candle['low']
                
                if signal == 'long':
                    if low <= stop_loss:
                        exit_type = 'stop_loss'
                        exit_price = stop_loss
                        exit_idx = df_trade.index[j]
                        break
                    elif high >= take_profit:
                        exit_type = 'take_profit'
                        exit_price = take_profit
                        exit_idx = df_trade.index[j]
                        break
                else:  # short
                    if high >= stop_loss:
                        exit_type = 'stop_loss'
                        exit_price = stop_loss
                        exit_idx = df_trade.index[j]
                        break
                    elif low <= take_profit:
                        exit_type = 'take_profit'
                        exit_price = take_profit
                        exit_idx = df_trade.index[j]
                        break
            
            if exit_type:
                if signal == 'long':
                    pnl = (exit_price - entry_price) * position_size * LEVERAGE
                else:
                    pnl = (entry_price - exit_price) * position_size * LEVERAGE
                
                balance += pnl
                
                trades.append({
                    'entry_time': idx,
                    'exit_time': exit_idx,
                    'side': signal,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'exit_type': exit_type,
                    'balance': balance
                })
                
                # Skip to exit candle
                i = df_trade.index.get_loc(exit_idx) + 1
                continue
        
        i += 1
    
    # Results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    
    if not trades:
        print("No trades executed!")
        return
    
    trades_df = pd.DataFrame(trades)
    
    total_trades = len(trades_df)
    winning = len(trades_df[trades_df['pnl'] > 0])
    losing = len(trades_df[trades_df['pnl'] < 0])
    win_rate = winning / total_trades * 100
    
    total_pnl = trades_df['pnl'].sum()
    avg_pnl = trades_df['pnl'].mean()
    
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing > 0 else 0
    
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Max drawdown
    balance_series = pd.Series([initial_balance] + trades_df['balance'].tolist())
    rolling_max = balance_series.expanding().max()
    drawdown = (balance_series - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()
    
    total_return = (balance - initial_balance) / initial_balance * 100
    
    sl_count = len(trades_df[trades_df['exit_type'] == 'stop_loss'])
    tp_count = len(trades_df[trades_df['exit_type'] == 'take_profit'])
    
    long_trades = trades_df[trades_df['side'] == 'long']
    short_trades = trades_df[trades_df['side'] == 'short']
    
    print(f"\nPeriod: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"Initial: ${initial_balance:,.2f}")
    print(f"Final: ${balance:,.2f}")
    print(f"Return: {total_return:+.2f}%")
    
    print(f"\n{'─'*40}")
    print(f"Total Trades: {total_trades}")
    print(f"  Winners: {winning} ({win_rate:.1f}%)")
    print(f"  Losers: {losing} ({100-win_rate:.1f}%)")
    
    print(f"\n{'─'*40}")
    print(f"Total P&L: ${total_pnl:+,.2f}")
    print(f"Avg P&L: ${avg_pnl:+,.2f}")
    print(f"Avg Win: ${avg_win:+,.2f}")
    print(f"Avg Loss: ${avg_loss:+,.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    
    print(f"\n{'─'*40}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    
    print(f"\n{'─'*40}")
    print(f"Take Profit: {tp_count} ({tp_count/total_trades*100:.1f}%)")
    print(f"Stop Loss: {sl_count} ({sl_count/total_trades*100:.1f}%)")
    
    print(f"\n{'─'*40}")
    print(f"Long: {len(long_trades)} trades | P&L: ${long_trades['pnl'].sum():+,.2f}")
    print(f"Short: {len(short_trades)} trades | P&L: ${short_trades['pnl'].sum():+,.2f}")
    
    # Save
    trades_df.to_csv('backtest_trades.csv', index=False)
    print(f"\nTrades saved to: backtest_trades.csv")
    
    # Yearly breakdown
    print(f"\n{'─'*40}")
    print("YEARLY BREAKDOWN:")
    trades_df['year'] = pd.to_datetime(trades_df['entry_time']).dt.year
    yearly = trades_df.groupby('year').agg({
        'pnl': ['sum', 'count', lambda x: (x > 0).sum() / len(x) * 100]
    }).round(2)
    yearly.columns = ['P&L', 'Trades', 'Win%']
    print(yearly.to_string())


if __name__ == "__main__":
    run_backtest()
