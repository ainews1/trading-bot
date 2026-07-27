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
from dotenv import load_dotenv

load_dotenv()

from config import config
from market_scout import MarketScout
import telegram_alert

# State file for persistence
STATE_FILE = "paper_state.json"

# Import strategy based on config
if config.STRATEGY == "bulldog":
    from strategy_bulldog import BulldogStrategy as Strategy, TradeSignal, Signal
elif config.STRATEGY == "donchian":
    from strategy_donchian import DonchianStrategy as Strategy, TradeSignal, Signal
elif config.STRATEGY == "vwap":
    from strategy_vwap import VWAPStrategy as Strategy, TradeSignal, Signal
elif config.STRATEGY == "orderflow":
    from strategy_orderflow import OrderFlowStrategy as Strategy, TradeSignal, Signal
elif config.STRATEGY == "multi":
    from strategy_multi import MultiStrategy as Strategy, TradeSignal, Signal
elif config.STRATEGY == "probability":
    from strategy_probability import ProbabilityStrategy as Strategy, TradeSignal, Signal
elif config.STRATEGY == "sqzmom_smc":
    from strategy_sqzmom_smc import SqzMomSmcStrategy as Strategy, TradeSignal, Signal
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
        elif config.STRATEGY == "donchian":
            from strategy_donchian import DonchianStrategy
            self.strategy = DonchianStrategy(
                risk_per_trade=config.RISK_PER_TRADE,
                leverage=config.LEVERAGE,
            )
        elif config.STRATEGY == "vwap":
            from strategy_vwap import VWAPStrategy
            self.strategy = VWAPStrategy(
                risk_per_trade=config.RISK_PER_TRADE,
                leverage=config.LEVERAGE,
            )
        elif config.STRATEGY == "orderflow":
            from strategy_orderflow import OrderFlowStrategy
            self.strategy = OrderFlowStrategy(
                risk_per_trade=config.RISK_PER_TRADE,
                leverage=config.LEVERAGE,
            )
        elif config.STRATEGY == "multi":
            from strategy_multi import MultiStrategy
            self.strategy = MultiStrategy(
                risk_per_trade=config.RISK_PER_TRADE,
                leverage=config.LEVERAGE,
                min_agreement=2,
            )
        elif config.STRATEGY == "probability":
            from strategy_probability import ProbabilityStrategy
            self.strategy = ProbabilityStrategy(
                risk_per_trade=config.RISK_PER_TRADE,
                leverage=config.LEVERAGE,
                long_threshold=60.0,   # 60% confidence to trade
                short_threshold=60.0,
            )
        elif config.STRATEGY == "sqzmom_smc":
            from strategy_sqzmom_smc import SqzMomSmcStrategy
            self.strategy = SqzMomSmcStrategy(
                bb_length=config.SQZ_BB_LENGTH,
                bb_mult=config.SQZ_BB_MULT,
                kc_length=config.SQZ_KC_LENGTH,
                kc_mult=config.SQZ_KC_MULT,
                mom_length=config.SQZ_MOM_LENGTH,
                sl_atr_mult=config.SQZ_SL_ATR_MULT,
                tp_atr_mult=config.SQZ_TP_ATR_MULT,
                htf_confluence=config.HTF_CONFLUENCE_ENABLED,
                risk_per_trade=config.RISK_PER_TRADE,
                leverage=config.LEVERAGE,
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
        self.daily_pnl = 0.0
        self.daily_pnl_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        self.max_daily_loss = config.MAX_DAILY_LOSS
        # Balance at the start of the trading day — the loss limit must be a
        # fixed fraction of this, not of the current (already-drawn-down) balance.
        self.daily_open_balance = 0.0

        # Paper trading state
        self.paper_balance = 1000.0  # Starting paper balance
        self.paper_position: Optional[Dict] = None
        self.paper_pnl = 0.0

        # Load persisted state if exists
        self._load_state()

        if self.paper_trading and self.daily_open_balance <= 0:
            # Not in saved state — reconstruct: opening balance = current minus today's PnL
            self.daily_open_balance = self.paper_balance - self.daily_pnl
        
        # Real position tracking
        self.current_position: Optional[Dict] = None
        
        # Market scouting system
        self.market_scout = MarketScout()
        self.last_scout_time: Optional[datetime] = None
        self.scout_interval = timedelta(minutes=5)
        self.last_snapshot = None  # most recent MarketSnapshot, used by regime gate

        # Prevent duplicate signals on the same candle after restart
        self._last_signal_candle: Optional[str] = None
        
        logger.info("=" * 60)
        logger.info("TRADING BOT INITIALIZED")
        logger.info(f"Mode: {'PAPER TRADING' if self.paper_trading else '[LIVE] LIVE TRADING'}")
        logger.info(f"Strategy: {config.STRATEGY.upper()}")
        logger.info(f"Symbol: {config.SYMBOL}")
        logger.info(f"Timeframe: {config.TIMEFRAME}")
        logger.info(f"Leverage: {config.LEVERAGE}x")
        logger.info("=" * 60)
        
        if not self.paper_trading:
            logger.warning("[!] LIVE TRADING MODE - REAL MONEY AT RISK [!]")
    
    def _save_state(self):
        """Save paper trading state to file"""
        if not self.paper_trading:
            return
        
        state = {
            'paper_balance': self.paper_balance,
            'paper_pnl': self.paper_pnl,
            'paper_position': self.paper_position,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_date': self.daily_pnl_date,
            'daily_open_balance': self.daily_open_balance,
            'updated_at': datetime.now(timezone.utc).isoformat()
        }
        
        try:
            tmp_file = STATE_FILE + '.tmp'
            with open(tmp_file, 'w') as f:
                json.dump(state, f, indent=2)
            os.replace(tmp_file, STATE_FILE)
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

            # Restore daily PnL tracking (reset if it's a new day)
            today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            saved_date = state.get('daily_pnl_date', '')
            if saved_date == today:
                self.daily_pnl = state.get('daily_pnl', 0.0)
                self.daily_pnl_date = saved_date
                self.daily_open_balance = state.get('daily_open_balance', 0.0)
            else:
                self.daily_pnl = 0.0
                self.daily_pnl_date = today
                self.daily_open_balance = self.paper_balance
                logger.info(f"  New trading day - daily PnL reset")

            logger.info(f"[STATE] Loaded saved state:")
            logger.info(f"  Balance: ${self.paper_balance:.2f}")
            logger.info(f"  Total PnL: ${self.paper_pnl:.2f}")
            logger.info(f"  Daily PnL: ${self.daily_pnl:.2f}")
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
    
    @staticmethod
    def _timeframe_seconds() -> int:
        """Parse config.TIMEFRAME into seconds (e.g. '5m' -> 300)."""
        tf = config.TIMEFRAME
        if tf.endswith('m'):
            return int(tf[:-1]) * 60
        if tf.endswith('h'):
            return int(tf[:-1]) * 3600
        if tf.endswith('d'):
            return int(tf[:-1]) * 86400
        return 300  # fallback

    def fetch_ohlcv(self, limit: int = 400) -> Optional[pd.DataFrame]:
        # 400 4h candles ≈ 66 days — enough warmup for the strategy's daily EMA20
        # confluence filter (EMA residual weight < 0.2% at that depth).
        """Fetch OHLCV candle data with retries. Returns CLOSED candles only."""
        for attempt in range(3):
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

                # Poloniex includes the still-forming candle as the last row
                # (verified via live probe). Drop it: indicators and wick-based
                # SL/TP checks must only ever see closed candles, matching
                # backtest semantics.
                interval_secs = self._timeframe_seconds()
                current_open = pd.Timestamp(
                    (int(time.time()) // interval_secs) * interval_secs, unit='s'
                )
                if len(df) > 0 and df.index[-1] >= current_open:
                    df = df.iloc[:-1]

                if len(df) == 0:
                    logger.error("Empty OHLCV response after dropping partial candle")
                    return None

                logger.debug(f"Fetched {len(df)} closed candles")
                return df

            except ccxt.RateLimitExceeded:
                logger.warning("Rate limit exceeded, waiting 60s...")
                time.sleep(60)
            except ccxt.NetworkError as e:
                wait = 10 * (attempt + 1)
                logger.warning(f"Network error (attempt {attempt+1}/3), retrying in {wait}s: {e}")
                time.sleep(wait)
            except Exception as e:
                logger.error(f"Error fetching OHLCV: {e}")
                return None

        logger.error("Failed to fetch OHLCV after 3 retries")
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
    
    def _cancel_remaining_orders(self):
        """Cancel remaining SL/TP orders after position closes (OCO behavior)"""
        if not self.current_position:
            return
        for key in ('sl_order_id', 'tp_order_id'):
            order_id = self.current_position.get(key)
            if order_id:
                try:
                    self.exchange.cancel_order(order_id, config.SYMBOL)
                    logger.info(f"Cancelled {key}: {order_id}")
                except Exception as e:
                    logger.debug(f"Could not cancel {key} {order_id}: {e}")

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
        
        # Save state FIRST — prevents duplicate opens on restart
        self._save_state()

        logger.info(f"[PAPER] Opened {side.upper()} position:")
        logger.info(f"  Entry: {signal.entry_price:.2f}")
        logger.info(f"  Size: {signal.position_size:.6f} BTC")
        logger.info(f"  Stop Loss: {signal.stop_loss:.2f}")
        logger.info(f"  Take Profit: {signal.take_profit:.2f}")

        telegram_alert.trade_opened(
            side, signal.entry_price, signal.position_size,
            signal.stop_loss, signal.take_profit, self.paper_balance,
            paper=True,
        )
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

            # Store order IDs for OCO cancellation
            self.current_position = {
                'side': side,
                'sl_order_id': sl_order['id'],
                'tp_order_id': tp_order['id'],
                'entry_price': signal.entry_price,
                'size': signal.position_size,
            }

            balance = self.get_account_balance()
            telegram_alert.trade_opened(
                side, signal.entry_price, signal.position_size,
                signal.stop_loss, signal.take_profit, balance,
                paper=False,
            )

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
    
    def check_paper_exit(self, current_price: float, candle_high: float = 0.0, candle_low: float = 0.0):
        """Check if paper position should be closed using candle high/low for wick detection"""
        if not self.paper_position:
            return

        pos = self.paper_position
        check_high = candle_high if candle_high is not None else current_price
        check_low = candle_low if candle_low is not None else current_price

        if pos['side'] == 'buy':
            # Long position — SL triggered by low, TP triggered by high
            if check_low <= pos['stop_loss']:
                pnl = (pos['stop_loss'] - pos['entry_price']) * pos['size']
                self._close_paper_position(pnl, "STOP LOSS")
            elif check_high >= pos['take_profit']:
                pnl = (pos['take_profit'] - pos['entry_price']) * pos['size']
                self._close_paper_position(pnl, "TAKE PROFIT")
        else:
            # Short position — SL triggered by high, TP triggered by low
            if check_high >= pos['stop_loss']:
                pnl = (pos['entry_price'] - pos['stop_loss']) * pos['size']
                self._close_paper_position(pnl, "STOP LOSS")
            elif check_low <= pos['take_profit']:
                pnl = (pos['entry_price'] - pos['take_profit']) * pos['size']
                self._close_paper_position(pnl, "TAKE PROFIT")
    
    def _close_paper_position(self, pnl: float, reason: str):
        """Close paper position and update balance"""
        self.paper_balance += pnl
        self.paper_pnl += pnl
        self.daily_pnl += pnl

        # Notify strategy of trade result for cooldown tracking
        if hasattr(self.strategy, 'record_trade_result'):
            self.strategy.record_trade_result(won=(pnl > 0))

        # Clear position and save state FIRST — prevents duplicate closes on restart
        self.paper_position = None
        self._save_state()

        emoji = "[WIN]" if pnl > 0 else "[LOSS]"
        logger.info(f"[PAPER] Position closed - {reason} {emoji}")
        logger.info(f"  PnL: ${pnl:.2f}")
        logger.info(f"  Daily PnL: ${self.daily_pnl:.2f}")
        logger.info(f"  Total PnL: ${self.paper_pnl:.2f}")
        logger.info(f"  Balance: ${self.paper_balance:.2f}")

        telegram_alert.trade_closed(
            pnl, reason, self.paper_balance,
            self.daily_pnl, self.paper_pnl, paper=True,
        )
    
    def _daily_loss_pct(self, balance: float) -> float:
        """Today's loss as a fraction of the day-OPENING balance.

        Dividing by the current balance would shrink the denominator as losses
        mount, letting the total daily loss exceed the configured cap.
        """
        if self.daily_pnl >= 0:
            return 0.0
        denom = self.daily_open_balance if self.daily_open_balance > 0 else balance
        if denom <= 0:
            return 0.0
        return -self.daily_pnl / denom

    def _regime_gate(self, signal) -> Optional[str]:
        """Return a block reason if this signal should be filtered by market regime, else None.

        Uses the latest MarketScout snapshot. Fails open: if filtering is disabled
        or no snapshot is available yet, never blocks.
        """
        if not getattr(config, "REGIME_FILTER_ENABLED", False):
            return None
        snap = self.last_snapshot
        if snap is None:
            return None

        side = signal.signal.value  # "LONG" or "SHORT"

        # Block entries in non-directional (ranging) markets — worst-performing bucket.
        if getattr(config, "BLOCK_RANGING_REGIME", False) and snap.market_regime == "RANGING":
            return f"RANGING regime (trend {snap.trend_direction} {snap.trend_strength:.2f})"

        # Block counter-trend entries: LONG against BEAR, SHORT against BULL.
        if getattr(config, "BLOCK_COUNTER_TREND", False):
            min_str = getattr(config, "COUNTER_TREND_MIN_STRENGTH", 0.0)
            if snap.trend_strength >= min_str:
                if side == "LONG" and snap.trend_direction == "BEAR":
                    return f"counter-trend LONG vs BEAR (strength {snap.trend_strength:.2f})"
                if side == "SHORT" and snap.trend_direction == "BULL":
                    return f"counter-trend SHORT vs BULL (strength {snap.trend_strength:.2f})"
        return None

    def wait_for_next_candle(self):
        """Wait for the next candle close without time drift"""
        now = datetime.now(timezone.utc)

        # Calculate next interval boundary using epoch math (avoids hour=24 bug)
        epoch = now.timestamp()
        interval_secs = self._timeframe_seconds()
        next_boundary = ((int(epoch) // interval_secs) + 1) * interval_secs
        wait_seconds = next_boundary - epoch

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
                
                # Market scouting - comprehensive market analysis
                now = datetime.now(timezone.utc)
                should_scout = (
                    self.last_scout_time is None or
                    (now - self.last_scout_time) >= self.scout_interval
                )
                
                if should_scout:
                    snapshot = self.market_scout.scout_market(df)
                    if snapshot:
                        report = self.market_scout.format_snapshot_report(snapshot)
                        logger.info("\n" + report)
                        self.last_scout_time = now
                        self.last_snapshot = snapshot
                
                logger.info(f"Current price: {current_price:.2f}")
                
                # Check paper position exits (use candle high/low for wick detection)
                if self.paper_trading:
                    candle_high = df['high'].iloc[-1]
                    candle_low = df['low'].iloc[-1]
                    self.check_paper_exit(current_price, candle_high, candle_low)
                
                # Reset daily PnL at the start of each new UTC day
                today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
                if today != self.daily_pnl_date:
                    logger.info(f"[DAILY RESET] New trading day ({today}). Daily PnL reset from ${self.daily_pnl:.2f} to $0.00")
                    self.daily_pnl = 0.0
                    self.daily_pnl_date = today
                    self.daily_open_balance = self.get_account_balance()
                    self._save_state()

                # Check daily loss limit
                balance = self.get_account_balance()
                if self.daily_open_balance <= 0:
                    self.daily_open_balance = balance
                if balance > 0:
                    daily_loss_pct = self._daily_loss_pct(balance)
                    if daily_loss_pct >= self.max_daily_loss:
                        logger.warning(f"[!] Daily loss limit reached ({daily_loss_pct:.1%}). Trading paused until next day.")
                        self.wait_for_next_candle()
                        continue
                else:
                    balance = 0

                # Skip if already in position
                if self.has_open_position():
                    logger.info("Position open, waiting for exit...")
                    self.wait_for_next_candle()
                    continue
                elif not self.paper_trading and self.current_position and self.current_position.get('sl_order_id'):
                    # Position closed — cancel remaining SL/TP orders (OCO)
                    self._cancel_remaining_orders()
                    # Track trade result for cooldown (approximate from balance change)
                    if hasattr(self.strategy, 'record_trade_result'):
                        # If we can't determine PnL, assume loss (conservative)
                        self.strategy.record_trade_result(won=False)
                    self.current_position = None
                
                # Get account balance
                balance = self.get_account_balance()
                logger.info(f"Account balance: ${balance:.2f}")
                
                # Analyze and generate signal
                signal = self.strategy.analyze(df, balance)

                # Regime gate: drop counter-trend / ranging-market entries
                if signal:
                    block_reason = self._regime_gate(signal)
                    if block_reason:
                        logger.info(f"[REGIME BLOCK] {signal.signal.value} entry filtered — {block_reason}")
                        signal = None

                if signal:
                    # Deduplicate: skip if we already traded on this candle
                    candle_key = str(df.index[-1])
                    if candle_key == self._last_signal_candle:
                        logger.info(f"[SKIP] Already traded on candle {candle_key}")
                    else:
                        logger.info(f"[SIGNAL] Signal detected: {signal.signal.value}")
                        success = self.execute_trade(signal)
                        if success:
                            self._last_signal_candle = candle_key
                            logger.info("Trade executed successfully")
                        else:
                            logger.error("Trade execution failed")
                else:
                    logger.info("No signal - conditions not met")
                
                # Wait for next candle
                self.wait_for_next_candle()
                self._consecutive_errors = 0

            except KeyboardInterrupt:
                logger.info("Shutdown requested...")
                break
            except Exception as e:
                logger.exception(f"Unexpected error: {e}")
                consecutive_errors = getattr(self, '_consecutive_errors', 0) + 1
                self._consecutive_errors = consecutive_errors
                if consecutive_errors >= 5:
                    logger.critical(f"Too many consecutive errors ({consecutive_errors}), shutting down")
                    break
                time.sleep(60)
        
        logger.info("Bot stopped.")
        if self.paper_trading:
            logger.info(f"Final paper balance: ${self.paper_balance:.2f}")
            logger.info(f"Total paper PnL: ${self.paper_pnl:.2f}")


if __name__ == "__main__":
    # Ensure only one instance runs at a time (cross-platform)
    import platform
    LOCK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".bot.lock")
    _lock_fh = open(LOCK_FILE, "w")
    try:
        if platform.system() == "Windows":
            import msvcrt
            msvcrt.locking(_lock_fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(_lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, IOError):
        print("ERROR: Another bot instance is already running. Exiting.")
        sys.exit(1)

    bot = TradingBot()
    bot.run()
