"""
Probability-Based Multi-Strategy
================================
Each strategy contributes a confidence score (0-100%)
Trade when combined probability exceeds threshold
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


class ProbabilityStrategy:
    """
    Scores each condition 0-100%, combines with weights
    Trades when total score exceeds threshold
    """
    
    def __init__(
        self,
        # Thresholds
        long_threshold: float = 60.0,   # Need 60%+ to go long
        short_threshold: float = 60.0,  # Need 60%+ to go short
        # Weights (must sum to 100)
        vwap_weight: float = 30.0,
        donchian_weight: float = 25.0,
        ema_weight: float = 25.0,
        rsi_weight: float = 20.0,
        # Risk params
        atr_period: int = 14,
        sl_atr_mult: float = 1.5,
        tp_atr_mult: float = 2.5,
        risk_per_trade: float = 0.01,
        leverage: int = 5,
    ):
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.vwap_weight = vwap_weight
        self.donchian_weight = donchian_weight
        self.ema_weight = ema_weight
        self.rsi_weight = rsi_weight
        self.atr_period = atr_period
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # ATR
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(self.atr_period).mean()
        
        # VWAP
        df['typical'] = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (df['typical'] * df['volume']).rolling(50).sum() / df['volume'].rolling(50).sum()
        df['vwap_dist'] = (df['close'] - df['vwap']) / df['vwap'] * 100
        
        # Donchian
        df['don_high'] = df['high'].rolling(20).max().shift(1)
        df['don_low'] = df['low'].rolling(20).min().shift(1)
        df['don_mid'] = (df['don_high'] + df['don_low']) / 2
        
        # EMAs
        df['ema8'] = df['close'].ewm(span=8, adjust=False).mean()
        df['ema24'] = df['close'].ewm(span=24, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))
        
        # Momentum
        df['mom'] = df['close'].pct_change(5) * 100
        
        return df
    
    def score_vwap(self, df: pd.DataFrame) -> tuple:
        """
        VWAP score: How far from VWAP + direction
        Returns: (long_score, short_score) 0-100
        """
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        vwap_dist = current['vwap_dist']
        bouncing = current['close'] > prev['close']
        falling = current['close'] < prev['close']
        
        long_score = 0
        short_score = 0
        
        # Below VWAP = potential long
        if vwap_dist < 0:
            # Score based on distance (more distance = higher score)
            dist_score = min(abs(vwap_dist) * 30, 70)  # Max 70 from distance
            if bouncing:
                dist_score += 30  # Bonus for bouncing
            long_score = dist_score
        
        # Above VWAP = potential short
        if vwap_dist > 0:
            dist_score = min(abs(vwap_dist) * 30, 70)
            if falling:
                dist_score += 30
            short_score = dist_score
        
        return min(long_score, 100), min(short_score, 100)
    
    def score_donchian(self, df: pd.DataFrame) -> tuple:
        """
        Donchian score: Position in channel + breakout proximity
        Returns: (long_score, short_score) 0-100
        """
        current = df.iloc[-1]
        
        price = current['close']
        high = current['don_high']
        low = current['don_low']
        mid = current['don_mid']
        
        if pd.isna(high) or pd.isna(low):
            return 0, 0
        
        channel_range = high - low
        if channel_range == 0:
            return 0, 0
        
        # Position in channel (0 = at low, 100 = at high)
        position = (price - low) / channel_range * 100
        
        long_score = 0
        short_score = 0
        
        # Near/above high = bullish breakout
        if position > 80:
            long_score = 50 + (position - 80) * 2.5  # 50-100
        elif position > 50:
            long_score = (position - 50) * 1.5  # 0-45
        
        # Near/below low = bearish breakout
        if position < 20:
            short_score = 50 + (20 - position) * 2.5  # 50-100
        elif position < 50:
            short_score = (50 - position) * 1.5  # 0-45
        
        return min(long_score, 100), min(short_score, 100)
    
    def score_ema(self, df: pd.DataFrame) -> tuple:
        """
        EMA score: Trend alignment + crossover recency
        Returns: (long_score, short_score) 0-100
        """
        current = df.iloc[-1]
        
        ema8 = current['ema8']
        ema24 = current['ema24']
        ema50 = current['ema50']
        price = current['close']
        
        long_score = 0
        short_score = 0
        
        # EMA alignment
        if ema8 > ema24:  # Bullish
            long_score += 40
            if ema24 > ema50:  # Strong bullish
                long_score += 20
            if price > ema8:  # Price above all
                long_score += 20
        else:  # Bearish
            short_score += 40
            if ema24 < ema50:
                short_score += 20
            if price < ema8:
                short_score += 20
        
        # Trend strength (gap between EMAs)
        gap_pct = abs(ema8 - ema24) / ema24 * 100
        strength_bonus = min(gap_pct * 10, 20)
        
        if ema8 > ema24:
            long_score += strength_bonus
        else:
            short_score += strength_bonus
        
        return min(long_score, 100), min(short_score, 100)
    
    def score_rsi(self, df: pd.DataFrame) -> tuple:
        """
        RSI score: Oversold/overbought + divergence
        Returns: (long_score, short_score) 0-100
        """
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        rsi = current['rsi']
        rsi_prev = prev['rsi']
        
        if pd.isna(rsi):
            return 0, 0
        
        long_score = 0
        short_score = 0
        
        # RSI levels
        if rsi < 30:  # Oversold = bullish
            long_score = 70 + (30 - rsi) * 1.5
        elif rsi < 45:
            long_score = (45 - rsi) * 2
        elif rsi > 70:  # Overbought = bearish
            short_score = 70 + (rsi - 70) * 1.5
        elif rsi > 55:
            short_score = (rsi - 55) * 2
        
        # RSI momentum
        if rsi > rsi_prev:  # Rising RSI
            long_score += 15
        else:  # Falling RSI
            short_score += 15
        
        return min(long_score, 100), min(short_score, 100)
    
    def calculate_position_size(self, balance: float, entry: float, stop_loss: float) -> float:
        risk_amount = balance * self.risk_per_trade
        price_risk = abs(entry - stop_loss)
        if price_risk == 0:
            return 0
        return risk_amount / price_risk
    
    def analyze(self, df: pd.DataFrame, account_balance: float) -> Optional[TradeSignal]:
        if len(df) < 60:
            return None
        
        df = self.calculate_indicators(df)
        
        current = df.iloc[-1]
        price = current['close']
        atr = current['atr']
        
        if pd.isna(atr):
            return None
        
        # Get scores from each strategy
        vwap_long, vwap_short = self.score_vwap(df)
        don_long, don_short = self.score_donchian(df)
        ema_long, ema_short = self.score_ema(df)
        rsi_long, rsi_short = self.score_rsi(df)
        
        # Calculate weighted totals
        long_score = (
            vwap_long * self.vwap_weight +
            don_long * self.donchian_weight +
            ema_long * self.ema_weight +
            rsi_long * self.rsi_weight
        ) / 100
        
        short_score = (
            vwap_short * self.vwap_weight +
            don_short * self.donchian_weight +
            ema_short * self.ema_weight +
            rsi_short * self.rsi_weight
        ) / 100
        
        # Log scores
        logger.info(f"[*] PROBABILITY SCORES:")
        logger.info(f"    VWAP:     L:{vwap_long:5.1f}% | S:{vwap_short:5.1f}% (weight: {self.vwap_weight}%)")
        logger.info(f"    Donchian: L:{don_long:5.1f}% | S:{don_short:5.1f}% (weight: {self.donchian_weight}%)")
        logger.info(f"    EMA:      L:{ema_long:5.1f}% | S:{ema_short:5.1f}% (weight: {self.ema_weight}%)")
        logger.info(f"    RSI:      L:{rsi_long:5.1f}% | S:{rsi_short:5.1f}% (weight: {self.rsi_weight}%)")
        logger.info(f"    ===================================")
        logger.info(f"    TOTAL:    L:{long_score:5.1f}% | S:{short_score:5.1f}% (threshold: {self.long_threshold}%)")
        
        # Check if ANY single indicator has >60% (strong conviction)
        single_long = max(vwap_long, don_long, ema_long, rsi_long)
        single_short = max(vwap_short, don_short, ema_short, rsi_short)
        
        if single_long >= 60:
            logger.info(f"    [!] Single indicator >60% LONG detected!")
        if single_short >= 60:
            logger.info(f"    [!] Single indicator >60% SHORT detected!")
        
        # Check thresholds - EITHER total OR single indicator >60%
        trigger_long = (long_score >= self.long_threshold) or (single_long >= 60)
        trigger_short = (short_score >= self.short_threshold) or (single_short >= 60)
        
        if trigger_long and long_score > short_score and single_long > single_short:
            stop_loss = price - (atr * self.sl_atr_mult)
            take_profit = price + (atr * self.tp_atr_mult)
            
            position_size = self.calculate_position_size(account_balance, price, stop_loss)
            
            logger.info(f"[+] LONG SIGNAL: {long_score:.1f}% confidence")
            
            return TradeSignal(
                signal=Signal.LONG,
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reason=f"PROB LONG: {long_score:.0f}% (V:{vwap_long:.0f} D:{don_long:.0f} E:{ema_long:.0f} R:{rsi_long:.0f})"
            )
        
        if trigger_short and short_score > long_score and single_short > single_long:
            stop_loss = price + (atr * self.sl_atr_mult)
            take_profit = price - (atr * self.tp_atr_mult)
            
            position_size = self.calculate_position_size(account_balance, price, stop_loss)
            
            logger.info(f"[-] SHORT SIGNAL: {short_score:.1f}% confidence")
            
            return TradeSignal(
                signal=Signal.SHORT,
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reason=f"PROB SHORT: {short_score:.0f}% (V:{vwap_short:.0f} D:{don_short:.0f} E:{ema_short:.0f} R:{rsi_short:.0f})"
            )
        
        return None


Strategy = ProbabilityStrategy
