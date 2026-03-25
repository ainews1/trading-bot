"""
Simple EMA Crossover Strategy
=============================
Proven, simple, profitable.

Rules:
- LONG: Fast EMA crosses above Slow EMA + price above both
- SHORT: Fast EMA crosses below Slow EMA + price below both
- Stop loss: ATR-based
- Take profit: 2x ATR (2:1 R:R)
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


class SimpleStrategy:
    """
    Simple EMA Crossover with ATR stops
    """
    
    def __init__(
        self,
        fast_ema: int = 9,
        slow_ema: int = 21,
        atr_period: int = 14,
        atr_multiplier_sl: float = 1.5,  # SL = 1.5x ATR
        atr_multiplier_tp: float = 3.0,  # TP = 3x ATR (2:1 R:R)
        risk_per_trade: float = 0.02,
        leverage: int = 5,
    ):
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.atr_period = atr_period
        self.atr_multiplier_sl = atr_multiplier_sl
        self.atr_multiplier_tp = atr_multiplier_tp
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage
    
    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(self.atr_period).mean()
    
    def calculate_position_size(
        self, 
        balance: float, 
        entry: float, 
        stop_loss: float
    ) -> float:
        """Calculate position size based on risk"""
        risk_amount = balance * self.risk_per_trade
        price_risk = abs(entry - stop_loss)
        if price_risk == 0:
            return 0
        return risk_amount / price_risk
    
    def analyze(
        self,
        df: pd.DataFrame,
        account_balance: float
    ) -> Optional[TradeSignal]:
        """
        Analyze for EMA crossover signals
        """
        min_periods = max(self.slow_ema, self.atr_period) + 5
        if len(df) < min_periods:
            return None
        
        # Calculate indicators
        df = df.copy()
        df['ema_fast'] = df['close'].ewm(span=self.fast_ema, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_ema, adjust=False).mean()
        df['atr'] = self.calculate_atr(df)
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        current_price = current['close']
        atr = current['atr']
        
        if pd.isna(atr) or atr == 0:
            return None
        
        # Check for crossover (within last 3 candles for flexibility)
        # Also check if fast EMA is trending in right direction
        ema_fast_rising = current['ema_fast'] > df.iloc[-5]['ema_fast']
        ema_fast_falling = current['ema_fast'] < df.iloc[-5]['ema_fast']
        
        # Recent crossover: fast crossed above slow in last 3 bars
        recent_cross_up = False
        recent_cross_down = False
        for k in range(-3, 0):
            if k-1 >= -len(df):
                if df.iloc[k-1]['ema_fast'] <= df.iloc[k-1]['ema_slow'] and df.iloc[k]['ema_fast'] > df.iloc[k]['ema_slow']:
                    recent_cross_up = True
                if df.iloc[k-1]['ema_fast'] >= df.iloc[k-1]['ema_slow'] and df.iloc[k]['ema_fast'] < df.iloc[k]['ema_slow']:
                    recent_cross_down = True
        
        fast_crossed_above = recent_cross_up and ema_fast_rising
        fast_crossed_below = recent_cross_down and ema_fast_falling
        
        # LONG signal
        if fast_crossed_above and current_price > current['ema_slow']:
            stop_loss = current_price - (atr * self.atr_multiplier_sl)
            take_profit = current_price + (atr * self.atr_multiplier_tp)
            
            position_size = self.calculate_position_size(
                account_balance, current_price, stop_loss
            )
            
            return TradeSignal(
                signal=Signal.LONG,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reason=f"EMA CROSS LONG: {self.fast_ema} crossed above {self.slow_ema}"
            )
        
        # SHORT signal
        if fast_crossed_below and current_price < current['ema_slow']:
            stop_loss = current_price + (atr * self.atr_multiplier_sl)
            take_profit = current_price - (atr * self.atr_multiplier_tp)
            
            position_size = self.calculate_position_size(
                account_balance, current_price, stop_loss
            )
            
            return TradeSignal(
                signal=Signal.SHORT,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reason=f"EMA CROSS SHORT: {self.fast_ema} crossed below {self.slow_ema}"
            )
        
        return None


# Alias
Strategy = SimpleStrategy
