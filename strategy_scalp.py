"""
5-Minute Scalping Strategy
==========================
Quick trades on RSI extremes with tight stops
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


class ScalpStrategy:
    """
    RSI Extreme Scalping
    - LONG: RSI < 25 + price bouncing
    - SHORT: RSI > 75 + price rejecting
    - Quick TP, tight SL
    """
    
    def __init__(
        self,
        rsi_period: int = 7,
        rsi_oversold: int = 25,
        rsi_overbought: int = 75,
        atr_period: int = 14,
        sl_atr_mult: float = 1.0,
        tp_atr_mult: float = 1.5,
        risk_per_trade: float = 0.01,
        leverage: int = 5,
        ema_period: int = 50,
    ):
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.atr_period = atr_period
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage
        self.ema_period = ema_period
    
    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
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
        if len(df) < 60:
            return None
        
        df = df.copy()
        df['rsi'] = self.calculate_rsi(df['close'])
        df['atr'] = self.calculate_atr(df)
        df['ema'] = df['close'].ewm(span=self.ema_period, adjust=False).mean()
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        
        price = current['close']
        rsi = current['rsi']
        atr = current['atr']
        ema = current['ema']
        
        if pd.isna(rsi) or pd.isna(atr):
            return None
        
        # LONG: RSI was oversold and now turning up
        rsi_oversold = prev['rsi'] < self.rsi_oversold
        rsi_turning_up = current['rsi'] > prev['rsi'] and prev['rsi'] < prev2['rsi']
        price_above_ema = price > ema
        
        if rsi_oversold and rsi_turning_up:
            stop_loss = price - (atr * self.sl_atr_mult)
            take_profit = price + (atr * self.tp_atr_mult)
            
            position_size = self.calculate_position_size(account_balance, price, stop_loss)
            
            return TradeSignal(
                signal=Signal.LONG,
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reason=f"SCALP LONG: RSI {prev['rsi']:.0f} turning up"
            )
        
        # SHORT: RSI was overbought and now turning down
        rsi_overbought = prev['rsi'] > self.rsi_overbought
        rsi_turning_down = current['rsi'] < prev['rsi'] and prev['rsi'] > prev2['rsi']
        price_below_ema = price < ema
        
        if rsi_overbought and rsi_turning_down:
            stop_loss = price + (atr * self.sl_atr_mult)
            take_profit = price - (atr * self.tp_atr_mult)
            
            position_size = self.calculate_position_size(account_balance, price, stop_loss)
            
            return TradeSignal(
                signal=Signal.SHORT,
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reason=f"SCALP SHORT: RSI {prev['rsi']:.0f} turning down"
            )
        
        return None


Strategy = ScalpStrategy
