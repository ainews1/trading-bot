"""
Market Profile Strategy (Dalton Auction Theory)
================================================
Based on "Markets in Profile" by Jim Dalton

Key Concepts:
- Value Area (VA): Where 70% of volume traded
- POC: Point of Control - highest volume price
- Trade when price leaves VA and fails = fade back to POC
- Trade breakouts when price accepts outside VA
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
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


class MarketProfileStrategy:
    """
    Auction-based trading using Market Profile concepts
    
    Strategy modes:
    1. FADE: Price rejects outside VA -> trade back to POC
    2. BREAKOUT: Price accepts outside VA -> trade continuation
    """
    
    def __init__(
        self,
        profile_period: int = 48,  # 48 x 5m = 4 hours for profile
        va_pct: float = 0.70,      # Value Area = 70% of volume
        price_bins: int = 50,      # Granularity of profile
        atr_period: int = 14,
        sl_atr_mult: float = 1.5,
        tp_to_poc: bool = True,    # Target POC or use ATR
        tp_atr_mult: float = 2.0,
        risk_per_trade: float = 0.01,
        leverage: int = 5,
        mode: str = 'fade',        # 'fade' or 'breakout'
    ):
        self.profile_period = profile_period
        self.va_pct = va_pct
        self.price_bins = price_bins
        self.atr_period = atr_period
        self.sl_atr_mult = sl_atr_mult
        self.tp_to_poc = tp_to_poc
        self.tp_atr_mult = tp_atr_mult
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage
        self.mode = mode
    
    def calculate_profile(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        """
        Calculate POC, VAH, VAL from price/volume data
        Returns: (poc, vah, val)
        """
        # Create price bins
        price_min = df['low'].min()
        price_max = df['high'].max()
        bins = np.linspace(price_min, price_max, self.price_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Distribute volume across price levels (TPO-like)
        volume_profile = np.zeros(self.price_bins)
        
        for _, row in df.iterrows():
            # Find which bins this candle touches
            low_bin = np.searchsorted(bins, row['low'], side='right') - 1
            high_bin = np.searchsorted(bins, row['high'], side='left')
            
            low_bin = max(0, min(low_bin, self.price_bins - 1))
            high_bin = max(0, min(high_bin, self.price_bins - 1))
            
            # Distribute volume across touched bins
            touched_bins = high_bin - low_bin + 1
            if touched_bins > 0:
                vol_per_bin = row['volume'] / touched_bins
                for b in range(low_bin, high_bin + 1):
                    if 0 <= b < self.price_bins:
                        volume_profile[b] += vol_per_bin
        
        # POC = bin with highest volume
        poc_idx = np.argmax(volume_profile)
        poc = bin_centers[poc_idx]
        
        # Value Area = 70% of total volume around POC
        total_vol = volume_profile.sum()
        target_vol = total_vol * self.va_pct
        
        # Expand from POC until we capture 70%
        va_vol = volume_profile[poc_idx]
        low_idx = poc_idx
        high_idx = poc_idx
        
        while va_vol < target_vol and (low_idx > 0 or high_idx < self.price_bins - 1):
            # Add the larger adjacent volume
            low_vol = volume_profile[low_idx - 1] if low_idx > 0 else 0
            high_vol = volume_profile[high_idx + 1] if high_idx < self.price_bins - 1 else 0
            
            if low_vol >= high_vol and low_idx > 0:
                low_idx -= 1
                va_vol += low_vol
            elif high_idx < self.price_bins - 1:
                high_idx += 1
                va_vol += high_vol
            else:
                break
        
        val = bin_centers[low_idx]  # Value Area Low
        vah = bin_centers[high_idx]  # Value Area High
        
        return poc, vah, val
    
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
        min_periods = self.profile_period + self.atr_period + 10
        if len(df) < min_periods:
            return None
        
        # Calculate profile from recent data
        profile_df = df.iloc[-self.profile_period-1:-1]  # Exclude current candle
        poc, vah, val = self.calculate_profile(profile_df)
        
        # Current candle
        df = df.copy()
        df['atr'] = self.calculate_atr(df)
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        
        price = current['close']
        atr = current['atr']
        
        if pd.isna(atr):
            return None
        
        # Position relative to value area
        above_va = price > vah
        below_va = price < val
        in_va = val <= price <= vah
        
        # Previous position
        prev_above_va = prev['close'] > vah
        prev_below_va = prev['close'] < val
        
        # FADE MODE: Trade rejections back to POC
        if self.mode == 'fade':
            # LONG: Price was below VA, now bouncing back in
            if prev_below_va and price > prev['close'] and price > val * 0.998:
                stop_loss = price - (atr * self.sl_atr_mult)
                
                if self.tp_to_poc:
                    take_profit = poc
                else:
                    take_profit = price + (atr * self.tp_atr_mult)
                
                if take_profit <= price:
                    return None
                
                position_size = self.calculate_position_size(account_balance, price, stop_loss)
                
                logger.info(f"🟢 PROFILE LONG: Below VA rejection, target POC {poc:.0f}")
                
                return TradeSignal(
                    signal=Signal.LONG,
                    entry_price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=position_size,
                    reason=f"PROFILE LONG: VAL reject, POC target {poc:.0f}"
                )
            
            # SHORT: Price was above VA, now falling back in
            if prev_above_va and price < prev['close'] and price < vah * 1.002:
                stop_loss = price + (atr * self.sl_atr_mult)
                
                if self.tp_to_poc:
                    take_profit = poc
                else:
                    take_profit = price - (atr * self.tp_atr_mult)
                
                if take_profit >= price:
                    return None
                
                position_size = self.calculate_position_size(account_balance, price, stop_loss)
                
                logger.info(f"🔴 PROFILE SHORT: Above VA rejection, target POC {poc:.0f}")
                
                return TradeSignal(
                    signal=Signal.SHORT,
                    entry_price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=position_size,
                    reason=f"PROFILE SHORT: VAH reject, POC target {poc:.0f}"
                )
        
        # BREAKOUT MODE: Trade acceptance outside VA
        elif self.mode == 'breakout':
            # LONG: Price breaking above VAH with momentum
            if above_va and prev_above_va and price > prev['close']:
                stop_loss = vah - (atr * 0.5)  # Stop just below VAH
                take_profit = price + (atr * self.tp_atr_mult)
                
                position_size = self.calculate_position_size(account_balance, price, stop_loss)
                
                return TradeSignal(
                    signal=Signal.LONG,
                    entry_price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=position_size,
                    reason=f"PROFILE BREAKOUT LONG: Accepted above VAH {vah:.0f}"
                )
            
            # SHORT: Price breaking below VAL with momentum
            if below_va and prev_below_va and price < prev['close']:
                stop_loss = val + (atr * 0.5)  # Stop just above VAL
                take_profit = price - (atr * self.tp_atr_mult)
                
                position_size = self.calculate_position_size(account_balance, price, stop_loss)
                
                return TradeSignal(
                    signal=Signal.SHORT,
                    entry_price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=position_size,
                    reason=f"PROFILE BREAKOUT SHORT: Accepted below VAL {val:.0f}"
                )
        
        return None


Strategy = MarketProfileStrategy
