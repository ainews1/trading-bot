"""
Bulldog Strategy Backtest
=========================
Tests the Bulldog pattern detection strategy on historical data
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import logging

# Suppress strategy logging for faster backtest
logging.getLogger('strategy_bulldog').setLevel(logging.WARNING)

from strategy_bulldog import BulldogStrategy, Signal
from config import config


def run_backtest():
    print("=" * 60)
    print("BULLDOG STRATEGY BACKTEST")
    print("=" * 60)
    
    # Load data
    print("\nLoading historical data...")
    df = pd.read_csv('historical_data.csv', index_col='timestamp', parse_dates=True)
    print(f"Loaded {len(df):,} candles")
    print(f"Period: {df.index[0]} to {df.index[-1]}")
    
    # Initialize strategy
    strategy = BulldogStrategy(
        lookback_period=config.BULLDOG_LOOKBACK,
        swing_lookback=config.BULLDOG_SWING_LOOKBACK,
        double_bottom_tolerance=config.BULLDOG_DOUBLE_BOTTOM_TOL,
        min_back_height=config.BULLDOG_MIN_BACK_HEIGHT,
        max_pullback_ratio=config.BULLDOG_MAX_PULLBACK,
        min_pullback_ratio=config.BULLDOG_MIN_PULLBACK,
        take_profit_fib_levels=config.BULLDOG_TP_FIB_LEVELS,
        entry_on_pullback=config.BULLDOG_ENTRY_ON_PULLBACK,
        entry_on_breakout=config.BULLDOG_ENTRY_ON_BREAKOUT,
    )
    
    # Backtest parameters
    initial_balance = 1000.0
    balance = initial_balance
    leverage = config.LEVERAGE
    
    trades = []
    patterns_found = 0
    
    # Need enough lookback for pattern detection
    lookback = config.BULLDOG_LOOKBACK + 20
    
    print(f"\nScanning for Bulldog patterns...")
    print(f"Lookback: {config.BULLDOG_LOOKBACK} candles")
    print(f"Double bottom tolerance: {config.BULLDOG_DOUBLE_BOTTOM_TOL*100:.1f}%")
    print(f"Pullback range: {config.BULLDOG_MIN_PULLBACK*100:.0f}% - {config.BULLDOG_MAX_PULLBACK*100:.1f}%")
    
    i = lookback
    last_progress = 0
    
    while i < len(df) - 50:
        # Progress indicator
        progress = int(i / len(df) * 100)
        if progress >= last_progress + 5:
            print(f"  Progress: {progress}% ({patterns_found} patterns found, {len(trades)} trades)")
            sys.stdout.flush()
            last_progress = progress
        
        # Get lookback window
        window = df.iloc[i-lookback:i+1].copy()
        
        # Check for signal
        signal = strategy.analyze(window, balance)
        
        if signal and signal.signal != Signal.NONE:
            patterns_found += 1
            entry_price = signal.entry_price
            stop_loss = signal.stop_loss
            take_profit = signal.take_profit
            position_size = signal.position_size
            
            is_long = signal.signal == Signal.LONG
            
            # Look for exit in next candles
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
                else:  # short
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
            
            # Timeout exit at last candle price
            if not exit_type:
                exit_type = 'timeout'
                exit_price = df.iloc[min(i + 99, len(df) - 1)]['close']
                exit_idx = df.index[min(i + 99, len(df) - 1)]
            
            # Calculate P&L
            if is_long:
                pnl = (exit_price - entry_price) * position_size * leverage
            else:
                pnl = (entry_price - exit_price) * position_size * leverage
            
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
    print("BACKTEST RESULTS")
    print("=" * 60)
    
    print(f"\nPatterns detected: {patterns_found}")
    
    if not trades:
        print("No trades executed!")
        return
    
    trades_df = pd.DataFrame(trades)
    
    total_trades = len(trades_df)
    winning = len(trades_df[trades_df['pnl'] > 0])
    losing = len(trades_df[trades_df['pnl'] < 0])
    win_rate = winning / total_trades * 100 if total_trades > 0 else 0
    
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
    timeout_count = len(trades_df[trades_df['exit_type'] == 'timeout'])
    
    long_trades = trades_df[trades_df['side'] == 'long']
    short_trades = trades_df[trades_df['side'] == 'short']
    
    # Save trades early in case of crash
    trades_df.to_csv('backtest_bulldog_trades.csv', index=False)
    
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
    print(f"Timeout: {timeout_count} ({timeout_count/total_trades*100:.1f}%)")
    
    print(f"\n{'─'*40}")
    print(f"Long: {len(long_trades)} trades | P&L: ${long_trades['pnl'].sum():+,.2f}")
    print(f"Short: {len(short_trades)} trades | P&L: ${short_trades['pnl'].sum():+,.2f}")
    
    # Save trades
    trades_df.to_csv('backtest_bulldog_trades.csv', index=False)
    print(f"\nTrades saved to: backtest_bulldog_trades.csv")
    
    # Show sample trades
    if len(trades_df) > 0:
        print(f"\n{'─'*40}")
        print("SAMPLE TRADES (first 5):")
        for _, trade in trades_df.head(5).iterrows():
            print(f"  {trade['entry_time']} | {trade['side'].upper():5} | "
                  f"Entry: ${trade['entry_price']:,.0f} | Exit: ${trade['exit_price']:,.0f} | "
                  f"P&L: ${trade['pnl']:+,.2f} ({trade['exit_type']})")


if __name__ == "__main__":
    run_backtest()
