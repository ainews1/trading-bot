"""
Donchian Breakout Strategy
==========================
Simple breakout trading - buy new highs, sell new lows.

Rules:
- LONG: Price breaks above highest high of last N candles
- SHORT: Price breaks below lowest low of last N candles
- Exit: Opposite signal or trailing stop
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


class BreakoutStrategy:
    """
    Donchian Channel Breakout
    """
    
    def __init__(
        self,
        channel_period: int = 20,  # 20 candles for channel
        atr_period: int = 14,
        sl_atr_mult: float = 2.0,
        tp_atr_mult: float = 4.0,  # 2:1 R:R
        risk_per_trade: float = 0.02,
        leverage: int = 5,
        trend_filter_period: int = 100,  # Only trade with trend
    ):
        self.channel_period = channel_period
        self.atr_period = atr_period
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage
        self.trend_filter_period = trend_filter_period
    
    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(self.atr_period).mean()
    
    def calculate_position_size(self, balance: float, entry: float, stop_loss: float) -> float:
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
        
        min_periods = max(self.channel_period, self.atr_period, self.trend_filter_period) + 5
        if len(df) < min_periods:
            return None
        
        df = df.copy()
        
        # Donchian Channel
        df['upper'] = df['high'].rolling(self.channel_period).max().shift(1)  # Shift to avoid lookahead
        df['lower'] = df['low'].rolling(self.channel_period).min().shift(1)
        df['atr'] = self.calculate_atr(df)
        
        # Trend filter: 100-period EMA
        df['trend_ema'] = df['close'].ewm(span=self.trend_filter_period, adjust=False).mean()
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        current_price = current['close']
        atr = current['atr']
        upper = current['upper']
        lower = current['lower']
        trend_ema = current['trend_ema']
        
        if pd.isna(atr) or pd.isna(upper) or pd.isna(lower):
            return None
        
        # Breakout detection
        broke_above = (prev['close'] <= prev['upper']) and (current['close'] > upper)
        broke_below = (prev['close'] >= prev['lower']) and (current['close'] < lower)
        
        # LONG: Break above upper channel + price above trend EMA
        if broke_above and current_price > trend_ema:
            stop_loss = current_price - (atr * self.sl_atr_mult)
            take_profit = current_price + (atr * self.tp_atr_mult)
            
            position_size = self.calculate_position_size(
                account_balance, current_price, stop_loss
            )
            
            return TradeSignal(
                signal=Signal.LONG,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reason=f"BREAKOUT LONG: Broke {self.channel_period}-bar high {upper:.0f}"
            )
        
        # SHORT: Break below lower channel + price below trend EMA
        if broke_below and current_price < trend_ema:
            stop_loss = current_price + (atr * self.sl_atr_mult)
            take_profit = current_price - (atr * self.tp_atr_mult)
            
            position_size = self.calculate_position_size(
                account_balance, current_price, stop_loss
            )
            
            return TradeSignal(
                signal=Signal.SHORT,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reason=f"BREAKOUT SHORT: Broke {self.channel_period}-bar low {lower:.0f}"
            )
        
        return None


Strategy = BreakoutStrategy
