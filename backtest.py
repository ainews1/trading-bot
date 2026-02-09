"""
Backtest Script - BTC/USDT Strategy
====================================
Tests the EMA + RSI strategy from 2022 to present
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional, List, Dict
import time

from config import config
from strategy import Strategy, Signal


class Backtest:
    def __init__(self):
        # Use Binance for historical data (better coverage)
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        self.strategy = Strategy(
            ema_period=config.EMA_PERIOD,
            rsi_period=config.RSI_PERIOD,
            rsi_oversold=config.RSI_OVERSOLD,
            rsi_overbought=config.RSI_OVERBOUGHT,
            stop_loss_pct=config.STOP_LOSS_PCT,
            take_profit_pct=config.TAKE_PROFIT_PCT,
            risk_per_trade=config.RISK_PER_TRADE,
            leverage=config.LEVERAGE,
            volume_multiplier=config.VOLUME_MULTIPLIER,
            volume_period=config.VOLUME_PERIOD
        )
        
        # Backtest parameters
        self.initial_balance = 1000.0
        self.balance = self.initial_balance
        self.leverage = config.LEVERAGE
        
        # Trade tracking
        self.trades: List[Dict] = []
        self.position: Optional[Dict] = None
        self.equity_curve: List[float] = []
        
    def fetch_historical_data(self, start_date: str = "2022-01-01") -> pd.DataFrame:
        """Fetch all historical data from start_date to now"""
        print(f"Fetching historical data from {start_date}...")
        
        # Use Binance symbol for backtest
        symbol = "BTC/USDT:USDT"
        
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        all_data = []
        current_ts = start_ts
        
        while current_ts < end_ts:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol,
                    config.TIMEFRAME,
                    since=current_ts,
                    limit=1500
                )
                
                if not ohlcv:
                    break
                    
                all_data.extend(ohlcv)
                current_ts = ohlcv[-1][0] + 1
                
                print(f"  Fetched {len(all_data)} candles... ({datetime.fromtimestamp(current_ts/1000).strftime('%Y-%m-%d')})")
                time.sleep(0.5)  # Rate limit
                
            except Exception as e:
                print(f"Error fetching data: {e}")
                time.sleep(5)
                continue
        
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset=['timestamp'])
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()
        
        print(f"Total candles: {len(df)}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        
        return df
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators for the full dataset"""
        df = df.copy()
        
        # EMA
        df['ema'] = df['close'].ewm(span=config.EMA_PERIOD, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=config.RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=config.RSI_PERIOD).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_prev'] = df['rsi'].shift(1)
        
        # Volume
        df['vol_avg'] = df['volume'].rolling(window=config.VOLUME_PERIOD).mean()
        
        return df
    
    def check_exit(self, row, next_rows: pd.DataFrame) -> tuple:
        """Check if position hits SL or TP in subsequent candles"""
        if not self.position:
            return None, None, None
        
        pos = self.position
        
        for idx, candle in next_rows.iterrows():
            high = candle['high']
            low = candle['low']
            
            if pos['side'] == 'long':
                # Check stop loss first (conservative)
                if low <= pos['stop_loss']:
                    return 'stop_loss', pos['stop_loss'], idx
                if high >= pos['take_profit']:
                    return 'take_profit', pos['take_profit'], idx
            else:  # short
                if high >= pos['stop_loss']:
                    return 'stop_loss', pos['stop_loss'], idx
                if low <= pos['take_profit']:
                    return 'take_profit', pos['take_profit'], idx
        
        return None, None, None
    
    def run_backtest(self, df: pd.DataFrame):
        """Run the backtest simulation"""
        print("\n" + "="*60)
        print("RUNNING BACKTEST")
        print("="*60)
        
        df = self.calculate_indicators(df)
        
        # Need at least 250 candles for indicators to warm up
        warmup = max(config.EMA_PERIOD, config.RSI_PERIOD, config.VOLUME_PERIOD) + 10
        
        i = warmup
        while i < len(df) - 1:
            row = df.iloc[i]
            current_price = row['close']
            
            # Track equity
            self.equity_curve.append(self.balance)
            
            # Check for exit if in position
            if self.position:
                exit_type, exit_price, exit_idx = self.check_exit(row, df.iloc[i+1:i+50])
                
                if exit_type:
                    self.close_position(exit_type, exit_price, exit_idx)
                    # Skip to exit candle
                    exit_pos = df.index.get_loc(exit_idx)
                    i = exit_pos + 1
                    continue
            
            # Check for entry signal if not in position
            if not self.position:
                # Create a slice for strategy analysis
                lookback_df = df.iloc[max(0, i-250):i+1].copy()
                signal = self.strategy.analyze(lookback_df, self.balance)
                
                if signal:
                    self.open_position(signal, df.index[i])
            
            i += 1
        
        # Close any remaining position at last price
        if self.position:
            last_price = df['close'].iloc[-1]
            self.close_position('end_of_data', last_price, df.index[-1])
        
        self.print_results(df)
    
    def open_position(self, signal, timestamp):
        """Open a new position"""
        side = 'long' if signal.signal == Signal.LONG else 'short'
        
        self.position = {
            'side': side,
            'entry_price': signal.entry_price,
            'size': signal.position_size,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'entry_time': timestamp
        }
    
    def close_position(self, exit_type: str, exit_price: float, exit_time):
        """Close position and record trade"""
        pos = self.position
        
        if pos['side'] == 'long':
            pnl = (exit_price - pos['entry_price']) * pos['size'] * self.leverage
        else:
            pnl = (pos['entry_price'] - exit_price) * pos['size'] * self.leverage
        
        self.balance += pnl
        
        self.trades.append({
            'entry_time': pos['entry_time'],
            'exit_time': exit_time,
            'side': pos['side'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'size': pos['size'],
            'pnl': pnl,
            'exit_type': exit_type,
            'balance_after': self.balance
        })
        
        self.position = None
    
    def print_results(self, df: pd.DataFrame):
        """Print backtest results"""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        
        if not self.trades:
            print("No trades executed!")
            return
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic stats
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Profit factor
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Max drawdown
        equity = pd.Series(self.equity_curve)
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        # Return
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        
        # Trade breakdown by type
        sl_trades = len(trades_df[trades_df['exit_type'] == 'stop_loss'])
        tp_trades = len(trades_df[trades_df['exit_type'] == 'take_profit'])
        
        # Long vs Short
        long_trades = trades_df[trades_df['side'] == 'long']
        short_trades = trades_df[trades_df['side'] == 'short']
        
        print(f"\nPeriod: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${self.balance:,.2f}")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"\n{'─'*40}")
        print(f"Total Trades: {total_trades}")
        print(f"  Winners: {winning_trades} ({win_rate:.1f}%)")
        print(f"  Losers: {losing_trades} ({100-win_rate:.1f}%)")
        print(f"\n{'─'*40}")
        print(f"Total P&L: ${total_pnl:+,.2f}")
        print(f"Average P&L: ${avg_pnl:+,.2f}")
        print(f"Average Win: ${avg_win:+,.2f}")
        print(f"Average Loss: ${avg_loss:+,.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"\n{'─'*40}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"\n{'─'*40}")
        print(f"Exit Breakdown:")
        print(f"  Take Profit: {tp_trades} ({tp_trades/total_trades*100:.1f}%)")
        print(f"  Stop Loss: {sl_trades} ({sl_trades/total_trades*100:.1f}%)")
        print(f"\n{'─'*40}")
        print(f"Long Trades: {len(long_trades)} | P&L: ${long_trades['pnl'].sum():+,.2f}")
        print(f"Short Trades: {len(short_trades)} | P&L: ${short_trades['pnl'].sum():+,.2f}")
        
        # Save trades to CSV
        trades_df.to_csv('backtest_trades.csv', index=False)
        print(f"\nTrade log saved to: backtest_trades.csv")
        
        # Monthly breakdown
        print(f"\n{'─'*40}")
        print("MONTHLY BREAKDOWN:")
        trades_df['month'] = pd.to_datetime(trades_df['entry_time']).dt.to_period('M')
        monthly = trades_df.groupby('month').agg({
            'pnl': ['sum', 'count'],
        }).round(2)
        monthly.columns = ['P&L', 'Trades']
        print(monthly.to_string())


if __name__ == "__main__":
    bt = Backtest()
    
    # Fetch data from 2022
    df = bt.fetch_historical_data("2022-01-01")
    
    # Save data for future use
    df.to_csv('historical_data.csv')
    print("Historical data saved to: historical_data.csv")
    
    # Run backtest
    bt.run_backtest(df)
