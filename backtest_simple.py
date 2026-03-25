"""
Simple Strategy Backtest
========================
"""

import pandas as pd
import numpy as np
import sys

from strategy_simple import SimpleStrategy, Signal


def run_backtest():
    print("=" * 60)
    print("SIMPLE EMA CROSSOVER BACKTEST")
    print("=" * 60)
    sys.stdout.flush()
    
    # Load data
    print("\nLoading historical data...")
    df = pd.read_csv('historical_data.csv', index_col='timestamp', parse_dates=True)
    print(f"Loaded {len(df):,} candles")
    print(f"Period: {df.index[0]} to {df.index[-1]}")
    sys.stdout.flush()
    
    # Initialize strategy - balanced EMAs
    strategy = SimpleStrategy(
        fast_ema=20,
        slow_ema=50,
        atr_period=14,
        atr_multiplier_sl=2.0,
        atr_multiplier_tp=3.0,
        risk_per_trade=0.01,
        leverage=3,
    )
    
    # Backtest parameters
    initial_balance = 1000.0
    balance = initial_balance
    
    trades = []
    lookback = 250  # Need more data for 50-period EMA
    
    print(f"\nStrategy: EMA {strategy.fast_ema}/{strategy.slow_ema} crossover")
    print(f"ATR SL: {strategy.atr_multiplier_sl}x | TP: {strategy.atr_multiplier_tp}x")
    print(f"Risk per trade: {strategy.risk_per_trade*100}%")
    print("\nScanning...")
    sys.stdout.flush()
    
    i = lookback
    last_progress = 0
    
    while i < len(df) - 100:
        # Progress
        progress = int(i / len(df) * 100)
        if progress >= last_progress + 10:
            print(f"  {progress}% - {len(trades)} trades, balance: ${balance:.2f}")
            sys.stdout.flush()
            last_progress = progress
        
        window = df.iloc[i-lookback:i+1].copy()
        signal = strategy.analyze(window, balance)
        
        if signal and signal.signal != Signal.NONE:
            entry_price = signal.entry_price
            stop_loss = signal.stop_loss
            take_profit = signal.take_profit
            position_size = signal.position_size
            is_long = signal.signal == Signal.LONG
            
            # Simulate trade
            exit_type = None
            exit_price = None
            exit_idx = None
            
            for j in range(i + 1, min(i + 100, len(df))):
                candle = df.iloc[j]
                high = candle['high']
                low = candle['low']
                
                if is_long:
                    if low <= stop_loss:
                        exit_type = 'stop_loss'
                        exit_price = stop_loss
                        exit_idx = df.index[j]
                        break
                    elif high >= take_profit:
                        exit_type = 'take_profit'
                        exit_price = take_profit
                        exit_idx = df.index[j]
                        break
                else:
                    if high >= stop_loss:
                        exit_type = 'stop_loss'
                        exit_price = stop_loss
                        exit_idx = df.index[j]
                        break
                    elif low <= take_profit:
                        exit_type = 'take_profit'
                        exit_price = take_profit
                        exit_idx = df.index[j]
                        break
            
            if not exit_type:
                exit_type = 'timeout'
                exit_price = df.iloc[min(i + 99, len(df) - 1)]['close']
                exit_idx = df.index[min(i + 99, len(df) - 1)]
            
            # Calculate P&L
            if is_long:
                pnl = (exit_price - entry_price) * position_size * strategy.leverage
            else:
                pnl = (entry_price - exit_price) * position_size * strategy.leverage
            
            balance += pnl
            
            trades.append({
                'entry_time': df.index[i],
                'exit_time': exit_idx,
                'side': 'long' if is_long else 'short',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'exit_price': exit_price,
                'pnl': pnl,
                'exit_type': exit_type,
                'balance': balance,
                'reason': signal.reason
            })
            
            # Skip past exit
            if exit_idx is not None:
                exit_loc = df.index.get_loc(exit_idx)
                i = exit_loc + 1
                continue
        
        i += 1
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if not trades:
        print("No trades!")
        return
    
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv('backtest_simple_trades.csv', index=False)
    
    total_trades = len(trades_df)
    winning = len(trades_df[trades_df['pnl'] > 0])
    losing = len(trades_df[trades_df['pnl'] <= 0])
    win_rate = winning / total_trades * 100 if total_trades > 0 else 0
    
    total_pnl = trades_df['pnl'].sum()
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing > 0 else 0
    
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Max drawdown
    balance_series = pd.Series([initial_balance] + trades_df['balance'].tolist())
    rolling_max = balance_series.expanding().max()
    drawdown = (balance_series - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()
    
    total_return = (balance - initial_balance) / initial_balance * 100
    
    sl_count = len(trades_df[trades_df['exit_type'] == 'stop_loss'])
    tp_count = len(trades_df[trades_df['exit_type'] == 'take_profit'])
    
    print(f"\nPeriod: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"Initial: ${initial_balance:,.2f}")
    print(f"Final: ${balance:,.2f}")
    print(f"Return: {total_return:+.2f}%")
    
    print(f"\nTotal Trades: {total_trades}")
    print(f"  Winners: {winning} ({win_rate:.1f}%)")
    print(f"  Losers: {losing}")
    
    print(f"\nTotal P&L: ${total_pnl:+,.2f}")
    print(f"Avg Win: ${avg_win:+,.2f}")
    print(f"Avg Loss: ${avg_loss:+,.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    
    print(f"\nTake Profit: {tp_count} | Stop Loss: {sl_count}")
    
    print(f"\nTrades saved to: backtest_simple_trades.csv")


if __name__ == "__main__":
    run_backtest()
