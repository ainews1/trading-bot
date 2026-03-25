"""
Aggressive EMA 8/24 Scalping Strategy
=====================================
Fast 5-minute scalping with tight EMAs
"""

import pandas as pd
import numpy as np
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


class FastEMAStrategy:
    """
    Aggressive EMA 8/24 Crossover Scalping
    - LONG: EMA8 crosses above EMA24
    - SHORT: EMA8 crosses below EMA24
    """
    
    def __init__(
        self,
        fast_ema: int = 8,
        slow_ema: int = 24,
        atr_period: int = 14,
        sl_atr_mult: float = 1.0,
        tp_atr_mult: float = 1.5,
        risk_per_trade: float = 0.01,
        leverage: int = 5,
    ):
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.atr_period = atr_period
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage
    
    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(self.atr_period).mean()
    
    def calculate_position_size(self, balance: float, entry: float, stop_loss: float) -> float:
        risk_amount = balance * self.risk_per_trade
        price_risk = abs(entry - stop_loss)
        if price_risk == 0:
            return 0
        return risk_amount / price_risk
    
    def analyze(self, df: pd.DataFrame, account_balance: float) -> Optional[TradeSignal]:
        if len(df) < 50:
            return None
        
        df = df.copy()
        df['ema8'] = df['close'].ewm(span=self.fast_ema, adjust=False).mean()
        df['ema24'] = df['close'].ewm(span=self.slow_ema, adjust=False).mean()
        df['atr'] = self.calculate_atr(df)
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        price = current['close']
        ema8 = current['ema8']
        ema24 = current['ema24']
        atr = current['atr']
        
        if pd.isna(atr):
            return None
        
        # Check for crossover
        cross_up = prev['ema8'] <= prev['ema24'] and current['ema8'] > current['ema24']
        cross_down = prev['ema8'] >= prev['ema24'] and current['ema8'] < current['ema24']
        
        # LONG: EMA8 crosses above EMA24
        if cross_up:
            stop_loss = price - (atr * self.sl_atr_mult)
            take_profit = price + (atr * self.tp_atr_mult)
            
            position_size = self.calculate_position_size(account_balance, price, stop_loss)
            
            logger.info(f"🟢 LONG SIGNAL: EMA8 crossed above EMA24 @ {price:.2f}")
            
            return TradeSignal(
                signal=Signal.LONG,
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reason=f"EMA8/24 LONG: Cross up @ {price:.0f}"
            )
        
        # SHORT: EMA8 crosses below EMA24
        if cross_down:
            stop_loss = price + (atr * self.sl_atr_mult)
            take_profit = price - (atr * self.tp_atr_mult)
            
            position_size = self.calculate_position_size(account_balance, price, stop_loss)
            
            logger.info(f"🔴 SHORT SIGNAL: EMA8 crossed below EMA24 @ {price:.2f}")
            
            return TradeSignal(
                signal=Signal.SHORT,
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reason=f"EMA8/24 SHORT: Cross down @ {price:.0f}"
            )
        
        return None


Strategy = FastEMAStrategy
