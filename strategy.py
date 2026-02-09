"""
Trading Strategy Module
=======================
EMA Trend Filter + RSI Mean Reversion Scalping Strategy
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class Signal(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


@dataclass
class TradeSignal:
    signal: Signal
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    reason: str


class Strategy:
    """
    EMA Trend Filter + RSI Mean Reversion Strategy (v2 - Community Improved)
    
    Long Entry: Price > EMA AND RSI crosses below 30 AND Volume > 1.5x avg
    Short Entry: Price < EMA AND RSI crosses above 70 AND Volume > 1.5x avg
    """
    
    def __init__(
        self,
        ema_period: int = 21,
        rsi_period: int = 14,
        rsi_oversold: int = 30,
        rsi_overbought: int = 70,
        stop_loss_pct: float = 0.012,
        take_profit_pct: float = 0.018,
        risk_per_trade: float = 0.02,
        leverage: int = 5,
        volume_multiplier: float = 1.5,
        volume_period: int = 20
    ):
        self.ema_period = ema_period
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage
        self.volume_multiplier = volume_multiplier
        self.volume_period = volume_period
        
        # Track previous RSI for crossover detection
        self.prev_rsi: Optional[float] = None
    
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float
    ) -> float:
        """
        Calculate position size based on 2% account risk
        
        Position Size = (Account Balance × Risk %) / |Entry - Stop Loss|
        Adjusted for leverage
        """
        risk_amount = account_balance * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk == 0:
            return 0
        
        # Base position size (in quote currency)
        base_size = risk_amount / (price_risk / entry_price)
        
        # Apply leverage
        position_size = base_size * self.leverage
        
        # Convert to contract size (BTC amount)
        contracts = position_size / entry_price
        
        return contracts
    
    def analyze(
        self,
        df: pd.DataFrame,
        account_balance: float
    ) -> Optional[TradeSignal]:
        """
        Analyze market data and generate trade signal
        
        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
            account_balance: Current account balance in USDT
            
        Returns:
            TradeSignal if conditions met, None otherwise
        """
        if len(df) < self.ema_period:
            logger.warning(f"Insufficient data: {len(df)} candles, need {self.ema_period}")
            return None
        
        # Calculate indicators
        close = df['close']
        volume = df['volume']
        ema = self.calculate_ema(close, self.ema_period)
        rsi = self.calculate_rsi(close, self.rsi_period)
        
        # Volume confirmation
        avg_volume = volume.rolling(window=self.volume_period).mean()
        current_volume = volume.iloc[-1]
        current_avg_volume = avg_volume.iloc[-1]
        volume_confirmed = current_volume > (current_avg_volume * self.volume_multiplier)
        
        # Current values
        current_price = close.iloc[-1]
        current_ema = ema.iloc[-1]
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2] if len(rsi) > 1 else current_rsi
        
        logger.info(
            f"Analysis: Price={current_price:.2f}, EMA={current_ema:.2f}, "
            f"RSI={current_rsi:.2f}, Prev RSI={prev_rsi:.2f}, "
            f"Vol={current_volume:.0f}, AvgVol={current_avg_volume:.0f}, VolOK={volume_confirmed}"
        )
        
        signal = Signal.NONE
        reason = ""
        
        # Long Signal: Price > EMA AND RSI crosses below 30 AND Volume confirmed
        if current_price > current_ema:
            if prev_rsi >= self.rsi_oversold and current_rsi < self.rsi_oversold:
                if volume_confirmed:
                    signal = Signal.LONG
                    reason = f"LONG: Price ({current_price:.2f}) > EMA ({current_ema:.2f}), RSI crossed below {self.rsi_oversold}, Volume confirmed"
                else:
                    logger.info(f"LONG signal blocked - volume not confirmed ({current_volume:.0f} < {current_avg_volume * self.volume_multiplier:.0f})")
        
        # Short Signal: Price < EMA AND RSI crosses above 70 AND Volume confirmed
        if current_price < current_ema:
            if prev_rsi <= self.rsi_overbought and current_rsi > self.rsi_overbought:
                if volume_confirmed:
                    signal = Signal.SHORT
                    reason = f"SHORT: Price ({current_price:.2f}) < EMA ({current_ema:.2f}), RSI crossed above {self.rsi_overbought}, Volume confirmed"
                else:
                    logger.info(f"SHORT signal blocked - volume not confirmed ({current_volume:.0f} < {current_avg_volume * self.volume_multiplier:.0f})")
        
        if signal == Signal.NONE:
            return None
        
        # Calculate stop loss and take profit
        if signal == Signal.LONG:
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)
        else:  # SHORT
            stop_loss = current_price * (1 + self.stop_loss_pct)
            take_profit = current_price * (1 - self.take_profit_pct)
        
        # Calculate position size
        position_size = self.calculate_position_size(
            account_balance,
            current_price,
            stop_loss
        )
        
        logger.info(f"Signal generated: {reason}")
        logger.info(
            f"Entry={current_price:.2f}, SL={stop_loss:.2f}, "
            f"TP={take_profit:.2f}, Size={position_size:.6f} BTC"
        )
        
        return TradeSignal(
            signal=signal,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            reason=reason
        )
