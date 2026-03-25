"""
Breakout Strategy Backtest
==========================
"""

import pandas as pd
import numpy as np
import sys

from strategy_breakout import BreakoutStrategy, Signal


def run_backtest():
    print("=" * 60)
    print("DONCHIAN BREAKOUT BACKTEST")
    print("=" * 60)
    sys.stdout.flush()
    
    print("\nLoading data...")
    df = pd.read_csv('historical_data.csv', index_col='timestamp', parse_dates=True)
    print(f"Loaded {len(df):,} candles")
    print(f"Period: {df.index[0]} to {df.index[-1]}")
    sys.stdout.flush()
    
    # Test different parameters
    configs = [
        {"channel": 20, "sl": 2.0, "tp": 4.0, "trend": 100},
        {"channel": 50, "sl": 2.0, "tp": 4.0, "trend": 200},
        {"channel": 20, "sl": 1.5, "tp": 3.0, "trend": 50},
    ]
    
    for cfg in configs:
        print(f"\n--- Testing: {cfg['channel']}-bar channel, SL:{cfg['sl']}x TP:{cfg['tp']}x, Trend:{cfg['trend']} ---")
        sys.stdout.flush()
        
        strategy = BreakoutStrategy(
            channel_period=cfg['channel'],
            atr_period=14,
            sl_atr_mult=cfg['sl'],
            tp_atr_mult=cfg['tp'],
            risk_per_trade=0.02,
            leverage=5,
            trend_filter_period=cfg['trend'],
        )
        
        initial_balance = 1000.0
        balance = initial_balance
        trades = []
        lookback = max(cfg['channel'], cfg['trend']) + 50
        
        i = lookback
        last_progress = 0
        
        while i < len(df) - 100 and balance > 10:
            progress = int(i / len(df) * 100)
            if progress >= last_progress + 25:
                print(f"  {progress}% - {len(trades)} trades, ${balance:.2f}")
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
                
                exit_type = None
                exit_price = None
                
                for j in range(i + 1, min(i + 200, len(df))):
                    candle = df.iloc[j]
                    
                    if is_long:
                        if candle['low'] <= stop_loss:
                            exit_type = 'stop_loss'
                            exit_price = stop_loss
                            break
                        elif candle['high'] >= take_profit:
                            exit_type = 'take_profit'
                            exit_price = take_profit
                            break
                    else:
                        if candle['high'] >= stop_loss:
                            exit_type = 'stop_loss'
                            exit_price = stop_loss
                            break
                        elif candle['low'] <= take_profit:
                            exit_type = 'take_profit'
                            exit_price = take_profit
                            break
                
                if not exit_type:
                    exit_type = 'timeout'
                    exit_price = df.iloc[min(i + 199, len(df) - 1)]['close']
                
                if is_long:
                    pnl = (exit_price - entry_price) * position_size * strategy.leverage
                else:
                    pnl = (entry_price - exit_price) * position_size * strategy.leverage
                
                balance += pnl
                trades.append({
                    'side': 'long' if is_long else 'short',
                    'pnl': pnl,
                    'exit_type': exit_type,
                    'balance': balance
                })
                
                i += 50  # Skip after trade
                continue
            
            i += 1
        
        # Results
        if trades:
            trades_df = pd.DataFrame(trades)
            total = len(trades_df)
            wins = len(trades_df[trades_df['pnl'] > 0])
            win_rate = wins / total * 100
            total_pnl = trades_df['pnl'].sum()
            ret = (balance - initial_balance) / initial_balance * 100
            
            tp = len(trades_df[trades_df['exit_type'] == 'take_profit'])
            sl = len(trades_df[trades_df['exit_type'] == 'stop_loss'])
            
            print(f"  RESULT: {total} trades | Win: {win_rate:.1f}% | P&L: ${total_pnl:+,.0f} | Return: {ret:+.1f}%")
            print(f"  TP: {tp} | SL: {sl}")
        else:
            print("  No trades")
        sys.stdout.flush()
    
    print("\n" + "=" * 60)
    print("DONE")


if __name__ == "__main__":
    run_backtest()
