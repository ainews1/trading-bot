"""
Multi-Strategy Trading System
=============================
Combines: VWAP + Donchian + EMA/RSI
Takes signals when multiple strategies agree
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict
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


class MultiStrategy:
    """
    Combines multiple strategies:
    - VWAP Mean Reversion (5m)
    - Donchian Breakout (uses longer lookback)
    - EMA/RSI Momentum
    
    Trade when 2+ strategies agree
    """
    
    def __init__(
        self,
        # VWAP params
        vwap_period: int = 50,
        vwap_distance: float = 1.0,
        # Donchian params
        donchian_period: int = 20,
        # EMA/RSI params
        ema_fast: int = 8,
        ema_slow: int = 24,
        rsi_period: int = 14,
        rsi_oversold: int = 35,
        rsi_overbought: int = 65,
        # Risk params
        atr_period: int = 14,
        sl_atr_mult: float = 1.5,
        tp_atr_mult: float = 2.5,
        risk_per_trade: float = 0.01,
        leverage: int = 5,
        min_agreement: int = 2,  # Min strategies that must agree
    ):
        self.vwap_period = vwap_period
        self.vwap_distance = vwap_distance
        self.donchian_period = donchian_period
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.atr_period = atr_period
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage
        self.min_agreement = min_agreement
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # ATR
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        df['atr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(self.atr_period).mean()
        
        # VWAP
        df['typical'] = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (df['typical'] * df['volume']).rolling(self.vwap_period).sum() / df['volume'].rolling(self.vwap_period).sum()
        df['vwap_dist'] = (df['close'] - df['vwap']) / df['vwap'] * 100
        
        # Donchian
        df['donchian_high'] = df['high'].rolling(self.donchian_period).max().shift(1)
        df['donchian_low'] = df['low'].rolling(self.donchian_period).min().shift(1)
        
        # EMA
        df['ema_fast'] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()
        df['ema_trend'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def check_vwap_signal(self, df: pd.DataFrame) -> str:
        """VWAP Mean Reversion signal"""
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        vwap_dist = current['vwap_dist']
        ema_trend = current['ema_trend']
        price = current['close']
        
        # LONG: Far below VWAP + bouncing + uptrend
        if (vwap_dist < -self.vwap_distance and 
            prev['vwap_dist'] < -self.vwap_distance and
            price > prev['close'] and 
            price > ema_trend):
            return "LONG"
        
        # SHORT: Far above VWAP + falling + downtrend
        if (vwap_dist > self.vwap_distance and 
            prev['vwap_dist'] > self.vwap_distance and
            price < prev['close'] and 
            price < ema_trend):
            return "SHORT"
        
        return "NONE"
    
    def check_donchian_signal(self, df: pd.DataFrame) -> str:
        """Donchian Breakout signal"""
        current = df.iloc[-1]
        
        price = current['close']
        high = current['donchian_high']
        low = current['donchian_low']
        ema_trend = current['ema_trend']
        
        # LONG: Break above channel + uptrend
        if price > high and price > ema_trend:
            return "LONG"
        
        # SHORT: Break below channel + downtrend
        if price < low and price < ema_trend:
            return "SHORT"
        
        return "NONE"
    
    def check_ema_rsi_signal(self, df: pd.DataFrame) -> str:
        """EMA crossover + RSI confirmation"""
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # EMA cross
        cross_up = prev['ema_fast'] <= prev['ema_slow'] and current['ema_fast'] > current['ema_slow']
        cross_down = prev['ema_fast'] >= prev['ema_slow'] and current['ema_fast'] < current['ema_slow']
        
        rsi = current['rsi']
        
        # LONG: Cross up + RSI not overbought
        if cross_up and rsi < self.rsi_overbought:
            return "LONG"
        
        # SHORT: Cross down + RSI not oversold
        if cross_down and rsi > self.rsi_oversold:
            return "SHORT"
        
        return "NONE"
    
    def calculate_position_size(self, balance: float, entry: float, stop_loss: float) -> float:
        risk_amount = balance * self.risk_per_trade
        price_risk = abs(entry - stop_loss)
        if price_risk == 0:
            return 0
        return risk_amount / price_risk
    
    def analyze(self, df: pd.DataFrame, account_balance: float) -> Optional[TradeSignal]:
        min_periods = max(self.vwap_period, self.donchian_period, 50) + 10
        if len(df) < min_periods:
            return None
        
        df = self.calculate_indicators(df)
        
        current = df.iloc[-1]
        price = current['close']
        atr = current['atr']
        
        if pd.isna(atr):
            return None
        
        # Get signals from each strategy
        vwap_signal = self.check_vwap_signal(df)
        donchian_signal = self.check_donchian_signal(df)
        ema_rsi_signal = self.check_ema_rsi_signal(df)
        
        signals = {
            'VWAP': vwap_signal,
            'Donchian': donchian_signal,
            'EMA/RSI': ema_rsi_signal
        }
        
        # Count agreements
        long_count = sum(1 for s in signals.values() if s == "LONG")
        short_count = sum(1 for s in signals.values() if s == "SHORT")
        
        # Log signals
        signal_str = " | ".join([f"{k}:{v}" for k, v in signals.items()])
        logger.info(f"[*] Multi-Strategy: {signal_str}")
        logger.info(f"    LONG votes: {long_count} | SHORT votes: {short_count} | Need: {self.min_agreement}")
        
        # LONG: Multiple strategies agree
        if long_count >= self.min_agreement:
            stop_loss = price - (atr * self.sl_atr_mult)
            take_profit = price + (atr * self.tp_atr_mult)
            
            position_size = self.calculate_position_size(account_balance, price, stop_loss)
            
            agreeing = [k for k, v in signals.items() if v == "LONG"]
            
            logger.info(f"[+] MULTI LONG SIGNAL: {', '.join(agreeing)} agree")
            
            return TradeSignal(
                signal=Signal.LONG,
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reason=f"MULTI LONG: {'+'.join(agreeing)} ({long_count}/{len(signals)})"
            )
        
        # SHORT: Multiple strategies agree
        if short_count >= self.min_agreement:
            stop_loss = price + (atr * self.sl_atr_mult)
            take_profit = price - (atr * self.tp_atr_mult)
            
            position_size = self.calculate_position_size(account_balance, price, stop_loss)
            
            agreeing = [k for k, v in signals.items() if v == "SHORT"]
            
            logger.info(f"[-] MULTI SHORT SIGNAL: {', '.join(agreeing)} agree")
            
            return TradeSignal(
                signal=Signal.SHORT,
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reason=f"MULTI SHORT: {'+'.join(agreeing)} ({short_count}/{len(signals)})"
            )
        
        return None


Strategy = MultiStrategy
