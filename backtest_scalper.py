"""
Scalper Strategy Backtest v4
=============================
Tests the trend pullback scalper on historical 5min BTC data.
"""

import pandas as pd
import numpy as np
import sys

from strategy_scalper import ScalperStrategy, Signal


def run_backtest(
    # Strategy params
    ema_fast=20,
    ema_slow=50,
    rsi_period=14,
    rsi_pullback_level=45,
    rsi_pullback_upper=55,
    setup_lookback=5,
    max_distance_from_ema=0.02,
    atr_period=14,
    atr_volatility_cap=2.5,
    swing_lookback=5,
    sl_buffer_pct=0.001,
    tp_rr_ratio=1.5,
    min_candles_between=8,
    risk_per_trade=0.015,
    leverage=5,
    # Backtest params
    initial_balance=1000.0,
    fee_pct=0.0006,
    max_hold_candles=30,
    data_file='historical_data.csv',
    verbose=True,
):
    if verbose:
        print("=" * 60)
        print("TREND PULLBACK SCALPER BACKTEST v4")
        print("=" * 60)
        sys.stdout.flush()

    df = pd.read_csv(data_file, index_col='timestamp', parse_dates=True)
    if verbose:
        print(f"Loaded {len(df):,} candles")
        print(f"Period: {df.index[0]} to {df.index[-1]}")
        sys.stdout.flush()

    strategy = ScalperStrategy(
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        rsi_period=rsi_period,
        rsi_pullback_level=rsi_pullback_level,
        rsi_pullback_upper=rsi_pullback_upper,
        setup_lookback=setup_lookback,
        max_distance_from_ema=max_distance_from_ema,
        atr_period=atr_period,
        atr_volatility_cap=atr_volatility_cap,
        atr_avg_period=50,
        swing_lookback=swing_lookback,
        sl_buffer_pct=sl_buffer_pct,
        tp_rr_ratio=tp_rr_ratio,
        min_candles_between=min_candles_between,
        risk_per_trade=risk_per_trade,
        leverage=leverage,
    )

    if verbose:
        print("Computing indicators...")
        sys.stdout.flush()
    df = strategy.compute_indicators(df)

    warmup = max(ema_slow, atr_period + 50) + 20
    balance = initial_balance
    trades = []
    last_trade_idx = -999
    equity_curve = [initial_balance]

    if verbose:
        print(f"Strategy: EMA({ema_fast}/{ema_slow}) + RSI({rsi_period}, pullback={rsi_pullback_level}/{rsi_pullback_upper})")
        print(f"SL: recent swing + {sl_buffer_pct*100:.1f}% (capped 2x ATR) | TP: {tp_rr_ratio}x risk")
        print(f"Fee: {fee_pct*100:.3f}% | Max hold: {max_hold_candles} candles | Risk: {risk_per_trade*100:.1f}%")
        print(f"\nScanning from candle {warmup}...")
        sys.stdout.flush()

    i = warmup
    last_progress = 0

    while i < len(df) - max_hold_candles - 1:
        if verbose:
            progress = int(i / len(df) * 100)
            if progress >= last_progress + 10:
                wr = 0
                if trades:
                    wr = sum(1 for t in trades if t['pnl'] > 0) / len(trades) * 100
                print(f"  {progress}% - {len(trades)} trades, WR: {wr:.1f}%, bal: ${balance:.2f}")
                sys.stdout.flush()
                last_progress = progress

        if i - last_trade_idx < min_candles_between:
            i += 1
            continue

        signal = strategy.check_signal(df, i)
        if signal == Signal.NONE:
            i += 1
            continue

        entry, sl, tp = strategy.get_levels(df, i, signal)
        position_size = strategy.calculate_position_size(balance, entry, sl)

        if position_size <= 0:
            i += 1
            continue

        is_long = signal == Signal.LONG
        entry_fee = position_size * entry * fee_pct

        exit_type = None
        exit_price = None
        exit_idx = None

        for j in range(i + 1, min(i + max_hold_candles + 1, len(df))):
            candle = df.iloc[j]
            if is_long:
                if candle['low'] <= sl:
                    exit_type, exit_price, exit_idx = 'stop_loss', sl, j
                    break
                if candle['high'] >= tp:
                    exit_type, exit_price, exit_idx = 'take_profit', tp, j
                    break
            else:
                if candle['high'] >= sl:
                    exit_type, exit_price, exit_idx = 'stop_loss', sl, j
                    break
                if candle['low'] <= tp:
                    exit_type, exit_price, exit_idx = 'take_profit', tp, j
                    break

        if not exit_type:
            exit_type = 'timeout'
            exit_j = min(i + max_hold_candles, len(df) - 1)
            exit_price = df.iloc[exit_j]['close']
            exit_idx = exit_j

        exit_fee = position_size * exit_price * fee_pct
        total_fees = entry_fee + exit_fee

        if is_long:
            raw_pnl = (exit_price - entry) * position_size
        else:
            raw_pnl = (entry - exit_price) * position_size

        pnl = raw_pnl - total_fees
        balance += pnl
        if balance < 0:
            balance = 0
        equity_curve.append(balance)

        trades.append({
            'entry_time': df.index[i],
            'exit_time': df.index[exit_idx],
            'side': 'long' if is_long else 'short',
            'entry_price': entry,
            'stop_loss': sl,
            'take_profit': tp,
            'exit_price': exit_price,
            'raw_pnl': raw_pnl,
            'fees': total_fees,
            'pnl': pnl,
            'exit_type': exit_type,
            'balance': balance,
            'rsi': df.iloc[i]['rsi'],
            'atr': df.iloc[i]['atr'],
        })

        last_trade_idx = exit_idx
        i = exit_idx + 1

    # Results
    if verbose:
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)

    if not trades:
        if verbose:
            print("No trades executed!")
        return None

    trades_df = pd.DataFrame(trades)
    trades_df.to_csv('backtest_scalper_trades.csv', index=False)

    total_trades = len(trades_df)
    winning = len(trades_df[trades_df['pnl'] > 0])
    losing = len(trades_df[trades_df['pnl'] <= 0])
    win_rate = winning / total_trades * 100 if total_trades > 0 else 0

    total_pnl = trades_df['pnl'].sum()
    total_fees_sum = trades_df['fees'].sum()
    avg_pnl = trades_df['pnl'].mean()
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing > 0 else 0

    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    eq = pd.Series(equity_curve)
    rolling_max = eq.expanding().max()
    dd = (eq - rolling_max) / rolling_max * 100
    max_drawdown = dd.min()

    total_return = (balance - initial_balance) / initial_balance * 100

    sl_count = len(trades_df[trades_df['exit_type'] == 'stop_loss'])
    tp_count = len(trades_df[trades_df['exit_type'] == 'take_profit'])
    timeout_count = len(trades_df[trades_df['exit_type'] == 'timeout'])

    long_trades = trades_df[trades_df['side'] == 'long']
    short_trades = trades_df[trades_df['side'] == 'short']
    long_wr = (long_trades['pnl'] > 0).mean() * 100 if len(long_trades) > 0 else 0
    short_wr = (short_trades['pnl'] > 0).mean() * 100 if len(short_trades) > 0 else 0

    if verbose:
        print(f"\nPeriod: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"Initial: ${initial_balance:,.2f}")
        print(f"Final: ${balance:,.2f}")
        print(f"Return: {total_return:+.2f}%")

        print(f"\n{'-'*50}")
        print(f"Total Trades: {total_trades}")
        print(f"  Winners: {winning} ({win_rate:.1f}%)")
        print(f"  Losers: {losing} ({100-win_rate:.1f}%)")

        print(f"\n{'-'*50}")
        print(f"Total P&L: ${total_pnl:+,.2f} (fees: ${total_fees_sum:,.2f})")
        print(f"Avg P&L: ${avg_pnl:+,.2f}")
        print(f"Avg Win: ${avg_win:+,.2f}")
        print(f"Avg Loss: ${avg_loss:+,.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")

        print(f"\n{'-'*50}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")

        print(f"\n{'-'*50}")
        print(f"Take Profit: {tp_count} ({tp_count/total_trades*100:.1f}%)")
        print(f"Stop Loss: {sl_count} ({sl_count/total_trades*100:.1f}%)")
        print(f"Timeout: {timeout_count} ({timeout_count/total_trades*100:.1f}%)")

        print(f"\n{'-'*50}")
        print(f"Long:  {len(long_trades):3} trades | WR: {long_wr:.1f}% | P&L: ${long_trades['pnl'].sum():+,.2f}")
        print(f"Short: {len(short_trades):3} trades | WR: {short_wr:.1f}% | P&L: ${short_trades['pnl'].sum():+,.2f}")

        print(f"\n{'-'*50}")
        print("YEARLY BREAKDOWN:")
        trades_df['year'] = pd.to_datetime(trades_df['entry_time']).dt.year
        for year, group in trades_df.groupby('year'):
            yr_wr = (group['pnl'] > 0).sum() / len(group) * 100
            print(f"  {year}: {len(group):3} trades | WR: {yr_wr:.1f}% | P&L: ${group['pnl'].sum():+,.2f}")

        print(f"\nTrades saved to: backtest_scalper_trades.csv")

    return {
        'win_rate': win_rate,
        'total_return': total_return,
        'total_trades': total_trades,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
    }


if __name__ == "__main__":
    run_backtest()
