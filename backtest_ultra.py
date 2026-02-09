"""
Ultra-Fast Backtest - Fully Vectorized
======================================
Uses numpy vectorization for maximum speed
"""

import pandas as pd
import numpy as np
import time

# Config
EMA_PERIOD = 200
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
STOP_LOSS_PCT = 0.015
TAKE_PROFIT_PCT = 0.03
LEVERAGE = 5
INITIAL_BALANCE = 1000.0


def run_backtest():
    start_time = time.time()
    
    print("Loading data...")
    df = pd.read_csv('historical_data.csv')
    print(f"Loaded {len(df):,} candles in {time.time()-start_time:.2f}s")
    
    # Convert to numpy for speed
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    print("Calculating indicators...")
    t0 = time.time()
    
    # EMA - vectorized
    ema = pd.Series(close).ewm(span=EMA_PERIOD, adjust=False).mean().values
    
    # RSI - vectorized
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(RSI_PERIOD).mean().values
    avg_loss = pd.Series(loss).rolling(RSI_PERIOD).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    rsi = np.nan_to_num(rsi, nan=50.0)
    
    print(f"Indicators done in {time.time()-t0:.2f}s")
    
    # Find signals - vectorized
    print("Finding signals...")
    t0 = time.time()
    
    rsi_prev = np.roll(rsi, 1)
    rsi_prev[0] = 50
    
    # Long: price > ema AND rsi crosses below 30
    long_signals = (close > ema) & (rsi_prev >= RSI_OVERSOLD) & (rsi < RSI_OVERSOLD)
    
    # Short: price < ema AND rsi crosses above 70  
    short_signals = (close < ema) & (rsi_prev <= RSI_OVERBOUGHT) & (rsi > RSI_OVERBOUGHT)
    
    # Get signal indices (skip warmup)
    warmup = max(EMA_PERIOD, RSI_PERIOD) + 10
    long_idx = np.where(long_signals[warmup:])[0] + warmup
    short_idx = np.where(short_signals[warmup:])[0] + warmup
    
    print(f"Found {len(long_idx)} long signals, {len(short_idx)} short signals in {time.time()-t0:.2f}s")
    
    # Simulate trades
    print("Simulating trades...")
    t0 = time.time()
    
    trades = []
    balance = INITIAL_BALANCE
    last_exit_idx = warmup
    
    # Combine and sort all signals
    all_signals = []
    for idx in long_idx:
        all_signals.append((idx, 'long'))
    for idx in short_idx:
        all_signals.append((idx, 'short'))
    all_signals.sort(key=lambda x: x[0])
    
    for sig_idx, side in all_signals:
        if sig_idx <= last_exit_idx:
            continue
        if balance <= 0:
            break
            
        entry_price = close[sig_idx]
        
        if side == 'long':
            sl = entry_price * (1 - STOP_LOSS_PCT)
            tp = entry_price * (1 + TAKE_PROFIT_PCT)
        else:
            sl = entry_price * (1 + STOP_LOSS_PCT)
            tp = entry_price * (1 - TAKE_PROFIT_PCT)
        
        # Position size (2% risk)
        risk_amount = balance * 0.02
        price_risk = abs(entry_price - sl)
        if price_risk == 0:
            continue
        pos_size = risk_amount / price_risk
        
        # Find exit (max 50 candles ahead)
        exit_type = None
        exit_price = None
        exit_idx = None
        
        end_idx = min(sig_idx + 50, len(close))
        
        for j in range(sig_idx + 1, end_idx):
            if side == 'long':
                if low[j] <= sl:
                    exit_type = 'sl'
                    exit_price = sl
                    exit_idx = j
                    break
                elif high[j] >= tp:
                    exit_type = 'tp'
                    exit_price = tp
                    exit_idx = j
                    break
            else:
                if high[j] >= sl:
                    exit_type = 'sl'
                    exit_price = sl
                    exit_idx = j
                    break
                elif low[j] <= tp:
                    exit_type = 'tp'
                    exit_price = tp
                    exit_idx = j
                    break
        
        if exit_type:
            if side == 'long':
                pnl = (exit_price - entry_price) * pos_size * LEVERAGE
            else:
                pnl = (entry_price - exit_price) * pos_size * LEVERAGE
            
            balance += pnl
            last_exit_idx = exit_idx
            
            trades.append({
                'idx': sig_idx,
                'side': side,
                'entry': entry_price,
                'exit': exit_price,
                'pnl': pnl,
                'exit_type': exit_type,
                'balance': balance
            })
    
    print(f"Simulation done in {time.time()-t0:.2f}s")
    
    # Results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    
    total_time = time.time() - start_time
    print(f"Total runtime: {total_time:.2f}s")
    
    if not trades:
        print("No trades!")
        return
    
    trades_df = pd.DataFrame(trades)
    
    total = len(trades_df)
    wins = len(trades_df[trades_df['pnl'] > 0])
    losses = len(trades_df[trades_df['pnl'] < 0])
    win_rate = wins / total * 100
    
    total_pnl = trades_df['pnl'].sum()
    
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else 0
    
    # Drawdown
    bal = [INITIAL_BALANCE] + trades_df['balance'].tolist()
    peak = np.maximum.accumulate(bal)
    dd = (np.array(bal) - peak) / peak * 100
    max_dd = dd.min()
    
    ret = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    
    sl_count = len(trades_df[trades_df['exit_type'] == 'sl'])
    tp_count = len(trades_df[trades_df['exit_type'] == 'tp'])
    
    longs = trades_df[trades_df['side'] == 'long']
    shorts = trades_df[trades_df['side'] == 'short']
    
    print(f"\nInitial: ${INITIAL_BALANCE:,.2f}")
    print(f"Final: ${balance:,.2f}")
    print(f"Return: {ret:+.2f}%")
    print(f"\nTrades: {total}")
    print(f"  Wins: {wins} ({win_rate:.1f}%)")
    print(f"  Losses: {losses}")
    print(f"\nTotal PnL: ${total_pnl:+,.2f}")
    print(f"Profit Factor: {pf:.2f}")
    print(f"Max Drawdown: {max_dd:.2f}%")
    print(f"\nTP exits: {tp_count} | SL exits: {sl_count}")
    print(f"Longs: {len(longs)} (${longs['pnl'].sum():+,.2f})")
    print(f"Shorts: {len(shorts)} (${shorts['pnl'].sum():+,.2f})")
    
    trades_df.to_csv('backtest_trades.csv', index=False)
    print("\nSaved to backtest_trades.csv")


if __name__ == "__main__":
    run_backtest()
