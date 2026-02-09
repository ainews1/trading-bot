"""
Poloniex Futures Trading Bot
============================
BTC/USDT Perpetual Scalping with EMA + RSI Strategy

⚠️ WARNING: Trading cryptocurrency futures involves substantial risk of loss.
Only trade with money you can afford to lose.
"""

import ccxt
import time
import logging
import pandas as pd
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import sys

from config import config

# State file for persistence
STATE_FILE = "paper_state.json"

# Import strategy based on config
if config.STRATEGY == "bulldog":
    from strategy_bulldog import BulldogStrategy as Strategy, TradeSignal, Signal
else:
    from strategy import Strategy, TradeSignal, Signal


# ===================
# Logging Setup
# ===================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TradingBot:
    """
    Production-ready trading bot for Poloniex Futures
    """
    
    def __init__(self):
        self.paper_trading = config.PAPER_TRADING
        self.exchange: Optional[ccxt.Exchange] = None
        
        # Initialize strategy based on config
        if config.STRATEGY == "bulldog":
            from strategy_bulldog import BulldogStrategy
            self.strategy = BulldogStrategy(
                lookback_period=config.BULLDOG_LOOKBACK,
                swing_lookback=config.BULLDOG_SWING_LOOKBACK,
                double_bottom_tolerance=config.BULLDOG_DOUBLE_BOTTOM_TOL,
                min_back_height=config.BULLDOG_MIN_BACK_HEIGHT,
                max_pullback_ratio=config.BULLDOG_MAX_PULLBACK,
                min_pullback_ratio=config.BULLDOG_MIN_PULLBACK,
                take_profit_fib_levels=config.BULLDOG_TP_FIB_LEVELS,
                risk_per_trade=config.RISK_PER_TRADE,
                leverage=config.LEVERAGE,
                entry_on_pullback=config.BULLDOG_ENTRY_ON_PULLBACK,
                entry_on_breakout=config.BULLDOG_ENTRY_ON_BREAKOUT,
            )
        else:
            from strategy import Strategy as EmaRsiStrategy
            self.strategy = EmaRsiStrategy(
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
        
        # Daily loss tracking
        self.daily_loss = 0.0
        self.max_daily_loss = config.MAX_DAILY_LOSS
        self.starting_balance = 1000.0
        
        # Paper trading state
        self.paper_balance = 1000.0  # Starting paper balance
        self.paper_position: Optional[Dict] = None
        self.paper_pnl = 0.0
        
        # Load persisted state if exists
        self._load_state()
        
        # Real position tracking
        self.current_position: Optional[Dict] = None
        
        logger.info("=" * 60)
        logger.info("TRADING BOT INITIALIZED")
        logger.info(f"Mode: {'PAPER TRADING' if self.paper_trading else '🔴 LIVE TRADING'}")
        logger.info(f"Strategy: {'🐕 BULLDOG' if config.STRATEGY == 'bulldog' else 'EMA+RSI'}")
        logger.info(f"Symbol: {config.SYMBOL}")
        logger.info(f"Timeframe: {config.TIMEFRAME}")
        logger.info(f"Leverage: {config.LEVERAGE}x")
        logger.info("=" * 60)
        
        if not self.paper_trading:
            logger.warning("⚠️  LIVE TRADING MODE - REAL MONEY AT RISK ⚠️")
    
    def _save_state(self):
        """Save paper trading state to file"""
        if not self.paper_trading:
            return
        
        state = {
            'paper_balance': self.paper_balance,
            'paper_pnl': self.paper_pnl,
            'paper_position': self.paper_position,
            'updated_at': datetime.now(timezone.utc).isoformat()
        }
        
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
            logger.debug(f"State saved to {STATE_FILE}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _load_state(self):
        """Load paper trading state from file"""
        if not self.paper_trading:
            return
        
        if not os.path.exists(STATE_FILE):
            logger.info("No saved state found, starting fresh")
            return
        
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
            
            self.paper_balance = state.get('paper_balance', 1000.0)
            self.paper_pnl = state.get('paper_pnl', 0.0)
            self.paper_position = state.get('paper_position')
            
            logger.info(f"[STATE] Loaded saved state:")
            logger.info(f"  Balance: ${self.paper_balance:.2f}")
            logger.info(f"  Total PnL: ${self.paper_pnl:.2f}")
            if self.paper_position:
                logger.info(f"  Open position: {self.paper_position['side'].upper()} @ {self.paper_position['entry_price']:.2f}")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
    
    def connect_exchange(self) -> bool:
        """Initialize exchange connection"""
        try:
            if self.paper_trading:
                # For paper trading, we still need exchange for price data
                self.exchange = ccxt.poloniex({
                    'enableRateLimit': True,
                    'options': {'defaultType': 'swap'}
                })
                logger.info("Connected to Poloniex (paper trading mode - read only)")
                return True
            
            if not config.API_KEY or not config.API_SECRET:
                logger.error("API credentials not set. Set POLONIEX_API_KEY and POLONIEX_API_SECRET")
                return False
            
            self.exchange = ccxt.poloniex({
                'apiKey': config.API_KEY,
                'secret': config.API_SECRET,
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'}
            })
            
            # Test connection
            balance = self.exchange.fetch_balance()
            logger.info(f"Connected to Poloniex. USDT Balance: {balance.get('USDT', {}).get('free', 0):.2f}")
            return True
            
        except ccxt.AuthenticationError as e:
            logger.error(f"Authentication failed: {e}")
            return False
        except ccxt.NetworkError as e:
            logger.error(f"Network error: {e}")
            return False
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def set_leverage(self) -> bool:
        """Set leverage to configured level"""
        if self.paper_trading:
            logger.info(f"[PAPER] Leverage set to {config.LEVERAGE}x")
            return True
        
        try:
            self.exchange.set_leverage(config.LEVERAGE, config.SYMBOL)
            logger.info(f"Leverage set to {config.LEVERAGE}x for {config.SYMBOL}")
            return True
        except ccxt.ExchangeError as e:
            logger.error(f"Failed to set leverage: {e}")
            return False
        except Exception as e:
            logger.error(f"Leverage error: {e}")
            return False
    
    def fetch_ohlcv(self, limit: int = 250) -> Optional[pd.DataFrame]:
        """Fetch OHLCV candle data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                config.SYMBOL,
                config.TIMEFRAME,
                limit=limit
            )
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.debug(f"Fetched {len(df)} candles")
            return df
            
        except ccxt.RateLimitExceeded:
            logger.warning("Rate limit exceeded, waiting 60s...")
            time.sleep(60)
            return None
        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching OHLCV: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching OHLCV: {e}")
            return None
    
    def get_account_balance(self) -> float:
        """Get current account balance"""
        if self.paper_trading:
            return self.paper_balance
        
        try:
            balance = self.exchange.fetch_balance()
            return float(balance.get('USDT', {}).get('free', 0))
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return 0.0
    
    def has_open_position(self) -> bool:
        """Check if there's an open position"""
        if self.paper_trading:
            return self.paper_position is not None
        
        try:
            positions = self.exchange.fetch_positions([config.SYMBOL])
            for pos in positions:
                if pos['symbol'] == config.SYMBOL and float(pos['contracts']) > 0:
                    self.current_position = pos
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking positions: {e}")
            return False
    
    def execute_trade(self, signal: TradeSignal) -> bool:
        """Execute a trade based on signal"""
        side = 'buy' if signal.signal == Signal.LONG else 'sell'
        
        if self.paper_trading:
            return self._paper_trade(signal, side)
        
        return self._live_trade(signal, side)
    
    def _paper_trade(self, signal: TradeSignal, side: str) -> bool:
        """Execute paper trade"""
        self.paper_position = {
            'side': side,
            'entry_price': signal.entry_price,
            'size': signal.position_size,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit
        }
        
        logger.info(f"[PAPER] Opened {side.upper()} position:")
        logger.info(f"  Entry: {signal.entry_price:.2f}")
        logger.info(f"  Size: {signal.position_size:.6f} BTC")
        logger.info(f"  Stop Loss: {signal.stop_loss:.2f}")
        logger.info(f"  Take Profit: {signal.take_profit:.2f}")
        
        self._save_state()
        return True
    
    def _live_trade(self, signal: TradeSignal, side: str) -> bool:
        """Execute live trade with stop loss and take profit"""
        try:
            # Place market order
            order = self.exchange.create_market_order(
                config.SYMBOL,
                side,
                signal.position_size
            )
            logger.info(f"Market order placed: {order['id']}")
            
            # Place stop loss
            sl_side = 'sell' if side == 'buy' else 'buy'
            sl_order = self.exchange.create_order(
                config.SYMBOL,
                'stop_market',
                sl_side,
                signal.position_size,
                None,
                {'stopPrice': signal.stop_loss}
            )
            logger.info(f"Stop loss placed at {signal.stop_loss:.2f}")
            
            # Place take profit
            tp_order = self.exchange.create_order(
                config.SYMBOL,
                'take_profit_market',
                sl_side,
                signal.position_size,
                None,
                {'stopPrice': signal.take_profit}
            )
            logger.info(f"Take profit placed at {signal.take_profit:.2f}")
            
            return True
            
        except ccxt.InsufficientFunds as e:
            logger.error(f"Insufficient funds: {e}")
            return False
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error: {e}")
            return False
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return False
    
    def check_paper_exit(self, current_price: float):
        """Check if paper position should be closed"""
        if not self.paper_position:
            return
        
        pos = self.paper_position
        
        if pos['side'] == 'buy':
            # Long position
            if current_price <= pos['stop_loss']:
                pnl = (pos['stop_loss'] - pos['entry_price']) * pos['size'] * config.LEVERAGE
                self._close_paper_position(pnl, "STOP LOSS")
            elif current_price >= pos['take_profit']:
                pnl = (pos['take_profit'] - pos['entry_price']) * pos['size'] * config.LEVERAGE
                self._close_paper_position(pnl, "TAKE PROFIT")
        else:
            # Short position
            if current_price >= pos['stop_loss']:
                pnl = (pos['entry_price'] - pos['stop_loss']) * pos['size'] * config.LEVERAGE
                self._close_paper_position(pnl, "STOP LOSS")
            elif current_price <= pos['take_profit']:
                pnl = (pos['entry_price'] - pos['take_profit']) * pos['size'] * config.LEVERAGE
                self._close_paper_position(pnl, "TAKE PROFIT")
    
    def _close_paper_position(self, pnl: float, reason: str):
        """Close paper position and update balance"""
        self.paper_balance += pnl
        self.paper_pnl += pnl
        
        emoji = "✅" if pnl > 0 else "❌"
        logger.info(f"[PAPER] Position closed - {reason} {emoji}")
        logger.info(f"  PnL: ${pnl:.2f}")
        logger.info(f"  Total PnL: ${self.paper_pnl:.2f}")
        logger.info(f"  Balance: ${self.paper_balance:.2f}")
        
        self.paper_position = None
        self._save_state()
    
    def wait_for_next_candle(self):
        """Wait for the next candle close without time drift"""
        now = datetime.now(timezone.utc)
        
        # Calculate seconds until next 5-minute mark
        minutes = now.minute
        
        # Next 5-minute interval
        next_interval = (minutes // 5 + 1) * 5
        if next_interval >= 60:
            # Roll over to next hour using timedelta (handles midnight correctly)
            next_time = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        else:
            next_time = now.replace(minute=next_interval, second=0, microsecond=0)
        
        wait_seconds = (next_time - now).total_seconds()
        
        # Add small buffer to ensure candle is closed
        wait_seconds += 2
        
        if wait_seconds > 0:
            logger.info(f"Waiting {wait_seconds:.0f}s for next candle...")
            time.sleep(wait_seconds)
    
    def run(self):
        """Main trading loop"""
        logger.info("Starting trading bot...")
        
        if not self.connect_exchange():
            logger.error("Failed to connect to exchange. Exiting.")
            return
        
        if not self.set_leverage():
            logger.error("Failed to set leverage. Exiting.")
            return
        
        logger.info("Bot running. Press Ctrl+C to stop.")
        
        while True:
            try:
                # Fetch latest data
                df = self.fetch_ohlcv()
                if df is None:
                    time.sleep(30)
                    continue
                
                current_price = df['close'].iloc[-1]
                logger.info(f"Current price: {current_price:.2f}")
                
                # Check paper position exits
                if self.paper_trading:
                    self.check_paper_exit(current_price)
                
                # Check daily loss limit
                if self.paper_trading:
                    daily_loss_pct = abs(self.paper_pnl) / self.starting_balance if self.paper_pnl < 0 else 0
                    if daily_loss_pct >= self.max_daily_loss:
                        logger.warning(f"⚠️ Daily loss limit reached ({daily_loss_pct:.1%}). Trading paused.")
                        self.wait_for_next_candle()
                        continue
                
                # Skip if already in position
                if self.has_open_position():
                    logger.info("Position open, waiting for exit...")
                    self.wait_for_next_candle()
                    continue
                
                # Get account balance
                balance = self.get_account_balance()
                logger.info(f"Account balance: ${balance:.2f}")
                
                # Analyze and generate signal
                signal = self.strategy.analyze(df, balance)
                
                if signal:
                    logger.info(f"🎯 Signal detected: {signal.signal.value}")
                    success = self.execute_trade(signal)
                    if success:
                        logger.info("Trade executed successfully")
                    else:
                        logger.error("Trade execution failed")
                else:
                    logger.info("No signal - conditions not met")
                
                # Wait for next candle
                self.wait_for_next_candle()
                
            except KeyboardInterrupt:
                logger.info("Shutdown requested...")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                time.sleep(60)
        
        logger.info("Bot stopped.")
        if self.paper_trading:
            logger.info(f"Final paper balance: ${self.paper_balance:.2f}")
            logger.info(f"Total paper PnL: ${self.paper_pnl:.2f}")


if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
