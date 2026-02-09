"""
Bulldog Pattern Trading Strategy
================================
Based on DejaBrewTrading's Bulldog Formation technique.

The Bulldog pattern is a double-bottom reversal pattern that looks like a dog:
- Back legs = First low
- Body/Back = Curved bounce up  
- Front legs = Second low (double bottom)
- Neck = Push up from double bottom
- Head = Shallow pullback (25-38.2% max)
- Entry = On head pullback or breakout

Works best on lower timeframes (1m, 5m) for scalping.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple
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


@dataclass
class BulldogPattern:
    """Detected Bulldog pattern data"""
    first_low_idx: int
    first_low_price: float
    back_high_idx: int
    back_high_price: float
    second_low_idx: int
    second_low_price: float
    neck_high_idx: int
    neck_high_price: float
    head_low_idx: int
    head_low_price: float
    pattern_strength: float  # 0-1 confidence score


class BulldogStrategy:
    """
    Bulldog Pattern Detection Strategy
    
    Detects the bulldog formation and generates entry signals.
    Also detects inverse bulldogs for SHORT entries.
    """
    
    def __init__(
        self,
        # Pattern detection params
        lookback_period: int = 50,          # Candles to look back for pattern
        swing_lookback: int = 5,            # Candles for swing high/low detection
        double_bottom_tolerance: float = 0.005,  # 0.5% tolerance for double bottom
        min_back_height: float = 0.005,     # Min 0.5% height for the "back"
        max_pullback_ratio: float = 0.382,  # Max 38.2% pullback for "head"
        min_pullback_ratio: float = 0.10,   # Min 10% pullback (not too shallow)
        
        # Risk management
        stop_loss_below_double_bottom: bool = True,
        stop_loss_fib_level: float = 0.618,  # If not below double bottom
        take_profit_fib_levels: List[float] = None,  # Extension targets
        risk_per_trade: float = 0.02,
        leverage: int = 5,
        
        # Entry style
        entry_on_pullback: bool = True,     # Enter on head pullback
        entry_on_breakout: bool = False,    # Or wait for breakout
    ):
        self.lookback_period = lookback_period
        self.swing_lookback = swing_lookback
        self.double_bottom_tolerance = double_bottom_tolerance
        self.min_back_height = min_back_height
        self.max_pullback_ratio = max_pullback_ratio
        self.min_pullback_ratio = min_pullback_ratio
        
        self.stop_loss_below_double_bottom = stop_loss_below_double_bottom
        self.stop_loss_fib_level = stop_loss_fib_level
        self.take_profit_fib_levels = take_profit_fib_levels or [0.5, 0.618, 1.0]
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage
        
        self.entry_on_pullback = entry_on_pullback
        self.entry_on_breakout = entry_on_breakout
    
    def find_swing_lows(self, df: pd.DataFrame, lookback: int = 5) -> List[int]:
        """Find swing low indices"""
        lows = df['low'].values
        swing_lows = []
        
        for i in range(lookback, len(lows) - lookback):
            if lows[i] == min(lows[i-lookback:i+lookback+1]):
                swing_lows.append(i)
        
        return swing_lows
    
    def find_swing_highs(self, df: pd.DataFrame, lookback: int = 5) -> List[int]:
        """Find swing high indices"""
        highs = df['high'].values
        swing_highs = []
        
        for i in range(lookback, len(highs) - lookback):
            if highs[i] == max(highs[i-lookback:i+lookback+1]):
                swing_highs.append(i)
        
        return swing_highs
    
    def is_downtrend(self, df: pd.DataFrame, start_idx: int, periods: int = 10) -> bool:
        """Check if there was a downtrend before the pattern"""
        if start_idx < periods:
            return False
        
        closes = df['close'].values[start_idx-periods:start_idx]
        # Simple check: price at start > price at end
        return closes[0] > closes[-1] * 1.005  # At least 0.5% drop
    
    def is_uptrend(self, df: pd.DataFrame, start_idx: int, periods: int = 10) -> bool:
        """Check if there was an uptrend before the pattern (for inverse)"""
        if start_idx < periods:
            return False
        
        closes = df['close'].values[start_idx-periods:start_idx]
        return closes[-1] > closes[0] * 1.005
    
    def calculate_retracement(self, low: float, high: float, current: float) -> float:
        """Calculate Fibonacci retracement level"""
        if high == low:
            return 0
        return (high - current) / (high - low)
    
    def detect_bulldog(self, df: pd.DataFrame) -> Optional[BulldogPattern]:
        """
        Detect bulldog pattern (for LONG entry)
        
        Pattern structure:
        1. Downtrend leading in
        2. First low (back legs)
        3. Curved bounce to resistance (back)
        4. Double bottom near first low (front legs)
        5. Push up (neck)
        6. Shallow pullback 25-38.2% (head)
        """
        if len(df) < self.lookback_period:
            return None
        
        # Work with recent data
        recent_df = df.iloc[-self.lookback_period:].copy()
        recent_df = recent_df.reset_index(drop=True)
        
        swing_lows = self.find_swing_lows(recent_df, self.swing_lookback)
        swing_highs = self.find_swing_highs(recent_df, self.swing_lookback)
        
        if len(swing_lows) < 2 or len(swing_highs) < 2:
            return None
        
        # Try to find bulldog pattern
        # We need: low1 -> high1 (back) -> low2 (near low1) -> high2 (neck) -> pullback
        
        for i, first_low_idx in enumerate(swing_lows[:-1]):
            first_low_price = recent_df['low'].iloc[first_low_idx]
            
            # Check for downtrend before first low
            if not self.is_downtrend(recent_df, first_low_idx):
                continue
            
            # Find the back (high after first low)
            back_candidates = [h for h in swing_highs if h > first_low_idx]
            if not back_candidates:
                continue
            
            for back_high_idx in back_candidates[:3]:  # Check first 3 highs
                back_high_price = recent_df['high'].iloc[back_high_idx]
                
                # Check back height (must be significant)
                back_height = (back_high_price - first_low_price) / first_low_price
                if back_height < self.min_back_height:
                    continue
                
                # Find second low (double bottom)
                second_low_candidates = [l for l in swing_lows if l > back_high_idx]
                if not second_low_candidates:
                    continue
                
                for second_low_idx in second_low_candidates[:2]:
                    second_low_price = recent_df['low'].iloc[second_low_idx]
                    
                    # Check double bottom tolerance
                    bottom_diff = abs(second_low_price - first_low_price) / first_low_price
                    if bottom_diff > self.double_bottom_tolerance:
                        continue
                    
                    # Find neck high after second low
                    neck_candidates = [h for h in swing_highs if h > second_low_idx]
                    if not neck_candidates:
                        continue
                    
                    neck_high_idx = neck_candidates[0]
                    neck_high_price = recent_df['high'].iloc[neck_high_idx]
                    
                    # Now check for the head (pullback)
                    # Look at candles after neck high
                    if neck_high_idx >= len(recent_df) - 3:
                        continue
                    
                    # Find the pullback low (head)
                    pullback_data = recent_df.iloc[neck_high_idx:]
                    if len(pullback_data) < 2:
                        continue
                    
                    head_low_idx_relative = pullback_data['low'].idxmin()
                    head_low_idx = head_low_idx_relative
                    head_low_price = recent_df['low'].iloc[head_low_idx]
                    
                    # Calculate pullback ratio
                    pullback_ratio = self.calculate_retracement(
                        second_low_price, neck_high_price, head_low_price
                    )
                    
                    # Check if pullback is in valid range (25-38.2%)
                    if pullback_ratio < self.min_pullback_ratio or pullback_ratio > self.max_pullback_ratio:
                        continue
                    
                    # Calculate pattern strength/confidence
                    strength = self._calculate_pattern_strength(
                        recent_df, first_low_idx, first_low_price,
                        back_high_idx, back_high_price,
                        second_low_idx, second_low_price,
                        neck_high_idx, neck_high_price,
                        head_low_idx, head_low_price,
                        pullback_ratio
                    )
                    
                    if strength > 0.5:  # Minimum confidence threshold
                        # Convert indices back to original dataframe
                        offset = len(df) - self.lookback_period
                        
                        return BulldogPattern(
                            first_low_idx=first_low_idx + offset,
                            first_low_price=first_low_price,
                            back_high_idx=back_high_idx + offset,
                            back_high_price=back_high_price,
                            second_low_idx=second_low_idx + offset,
                            second_low_price=second_low_price,
                            neck_high_idx=neck_high_idx + offset,
                            neck_high_price=neck_high_price,
                            head_low_idx=head_low_idx + offset,
                            head_low_price=head_low_price,
                            pattern_strength=strength
                        )
        
        return None
    
    def detect_inverse_bulldog(self, df: pd.DataFrame) -> Optional[BulldogPattern]:
        """
        Detect inverse bulldog pattern (for SHORT entry)
        Mirror of bulldog: double top with shallow bounce
        """
        if len(df) < self.lookback_period:
            return None
        
        recent_df = df.iloc[-self.lookback_period:].copy()
        recent_df = recent_df.reset_index(drop=True)
        
        swing_lows = self.find_swing_lows(recent_df, self.swing_lookback)
        swing_highs = self.find_swing_highs(recent_df, self.swing_lookback)
        
        if len(swing_lows) < 2 or len(swing_highs) < 2:
            return None
        
        # Inverse: high1 -> low1 (back) -> high2 (near high1) -> low2 (neck) -> pullback up
        
        for first_high_idx in swing_highs[:-1]:
            first_high_price = recent_df['high'].iloc[first_high_idx]
            
            # Check for uptrend before first high
            if not self.is_uptrend(recent_df, first_high_idx):
                continue
            
            # Find the back (low after first high)
            back_candidates = [l for l in swing_lows if l > first_high_idx]
            if not back_candidates:
                continue
            
            for back_low_idx in back_candidates[:3]:
                back_low_price = recent_df['low'].iloc[back_low_idx]
                
                # Check back depth
                back_depth = (first_high_price - back_low_price) / first_high_price
                if back_depth < self.min_back_height:
                    continue
                
                # Find second high (double top)
                second_high_candidates = [h for h in swing_highs if h > back_low_idx]
                if not second_high_candidates:
                    continue
                
                for second_high_idx in second_high_candidates[:2]:
                    second_high_price = recent_df['high'].iloc[second_high_idx]
                    
                    # Check double top tolerance
                    top_diff = abs(second_high_price - first_high_price) / first_high_price
                    if top_diff > self.double_bottom_tolerance:
                        continue
                    
                    # Find neck low
                    neck_candidates = [l for l in swing_lows if l > second_high_idx]
                    if not neck_candidates:
                        continue
                    
                    neck_low_idx = neck_candidates[0]
                    neck_low_price = recent_df['low'].iloc[neck_low_idx]
                    
                    # Check for head (pullback up)
                    if neck_low_idx >= len(recent_df) - 3:
                        continue
                    
                    pullback_data = recent_df.iloc[neck_low_idx:]
                    if len(pullback_data) < 2:
                        continue
                    
                    head_high_idx = pullback_data['high'].idxmax()
                    head_high_price = recent_df['high'].iloc[head_high_idx]
                    
                    # Calculate pullback ratio (inverted)
                    if second_high_price == neck_low_price:
                        continue
                    pullback_ratio = (head_high_price - neck_low_price) / (second_high_price - neck_low_price)
                    
                    if pullback_ratio < self.min_pullback_ratio or pullback_ratio > self.max_pullback_ratio:
                        continue
                    
                    strength = 0.6  # Simplified for inverse
                    
                    if strength > 0.5:
                        offset = len(df) - self.lookback_period
                        
                        # Return as BulldogPattern but inverted semantics
                        return BulldogPattern(
                            first_low_idx=first_high_idx + offset,  # Actually first high
                            first_low_price=first_high_price,
                            back_high_idx=back_low_idx + offset,    # Actually back low
                            back_high_price=back_low_price,
                            second_low_idx=second_high_idx + offset,  # Actually second high
                            second_low_price=second_high_price,
                            neck_high_idx=neck_low_idx + offset,    # Actually neck low
                            neck_high_price=neck_low_price,
                            head_low_idx=head_high_idx + offset,    # Actually head high
                            head_low_price=head_high_price,
                            pattern_strength=strength
                        )
        
        return None
    
    def _calculate_pattern_strength(
        self, df: pd.DataFrame,
        first_low_idx: int, first_low_price: float,
        back_high_idx: int, back_high_price: float,
        second_low_idx: int, second_low_price: float,
        neck_high_idx: int, neck_high_price: float,
        head_low_idx: int, head_low_price: float,
        pullback_ratio: float
    ) -> float:
        """Calculate confidence score for the pattern"""
        score = 0.0
        
        # 1. Double bottom quality (closer = better)
        bottom_diff = abs(second_low_price - first_low_price) / first_low_price
        if bottom_diff < 0.002:
            score += 0.25
        elif bottom_diff < 0.004:
            score += 0.15
        else:
            score += 0.05
        
        # 2. Pullback quality (25-30% is ideal)
        if 0.20 <= pullback_ratio <= 0.32:
            score += 0.25
        elif 0.15 <= pullback_ratio <= 0.382:
            score += 0.15
        else:
            score += 0.05
        
        # 3. Pattern timing (not too stretched out)
        pattern_length = head_low_idx - first_low_idx
        if pattern_length <= 30:
            score += 0.2
        elif pattern_length <= 45:
            score += 0.1
        
        # 4. Volume confirmation on second low (if available)
        try:
            vol_at_second_low = df['volume'].iloc[second_low_idx]
            avg_vol = df['volume'].iloc[first_low_idx:second_low_idx].mean()
            if vol_at_second_low > avg_vol * 1.2:
                score += 0.15
            else:
                score += 0.05
        except:
            score += 0.05
        
        # 5. Neck breaks above back
        if neck_high_price >= back_high_price * 0.995:
            score += 0.15
        else:
            score += 0.05
        
        return min(score, 1.0)
    
    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float
    ) -> float:
        """Calculate position size based on risk"""
        risk_amount = account_balance * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk == 0:
            return 0
        
        base_size = risk_amount / (price_risk / entry_price)
        position_size = base_size * self.leverage
        contracts = position_size / entry_price
        
        return contracts
    
    def analyze(
        self,
        df: pd.DataFrame,
        account_balance: float
    ) -> Optional[TradeSignal]:
        """
        Analyze for Bulldog pattern and generate trade signal
        """
        if len(df) < self.lookback_period:
            logger.warning(f"Insufficient data: {len(df)} candles, need {self.lookback_period}")
            return None
        
        current_price = df['close'].iloc[-1]
        
        # Try to detect bulldog (LONG)
        bulldog = self.detect_bulldog(df)
        if bulldog:
            logger.info(f"🐕 BULLDOG DETECTED! Strength: {bulldog.pattern_strength:.2f}")
            logger.info(f"  First low: {bulldog.first_low_price:.2f}")
            logger.info(f"  Back high: {bulldog.back_high_price:.2f}")
            logger.info(f"  Double bottom: {bulldog.second_low_price:.2f}")
            logger.info(f"  Neck high: {bulldog.neck_high_price:.2f}")
            logger.info(f"  Head low: {bulldog.head_low_price:.2f}")
            
            # Check entry conditions
            entry_valid = False
            
            if self.entry_on_pullback:
                # Entry on head pullback - price should be near head low and starting to rise
                recent_low = df['low'].iloc[-3:].min()
                price_near_head = abs(current_price - bulldog.head_low_price) / current_price < 0.01
                price_bouncing = current_price > recent_low
                entry_valid = price_near_head and price_bouncing
            
            if self.entry_on_breakout and not entry_valid:
                # Entry on breakout above neck high
                entry_valid = current_price > bulldog.neck_high_price * 1.001
            
            if entry_valid:
                # Calculate stop loss
                if self.stop_loss_below_double_bottom:
                    stop_loss = min(bulldog.first_low_price, bulldog.second_low_price) * 0.998
                else:
                    # Use 61.8% fib level
                    fib_range = bulldog.neck_high_price - bulldog.second_low_price
                    stop_loss = bulldog.neck_high_price - (fib_range * self.stop_loss_fib_level)
                
                # Calculate take profit using Fibonacci extensions
                # From lowest low to back high
                lowest_low = min(bulldog.first_low_price, bulldog.second_low_price)
                fib_base = bulldog.back_high_price - lowest_low
                take_profit = bulldog.back_high_price + (fib_base * self.take_profit_fib_levels[0])
                
                position_size = self.calculate_position_size(
                    account_balance, current_price, stop_loss
                )
                
                return TradeSignal(
                    signal=Signal.LONG,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=position_size,
                    reason=f"BULLDOG LONG: Pattern strength {bulldog.pattern_strength:.0%}, "
                           f"Double bottom @ {bulldog.second_low_price:.2f}"
                )
        
        # Try to detect inverse bulldog (SHORT)
        inverse = self.detect_inverse_bulldog(df)
        if inverse:
            logger.info(f"🐕 INVERSE BULLDOG DETECTED! Strength: {inverse.pattern_strength:.2f}")
            
            entry_valid = False
            
            if self.entry_on_pullback:
                recent_high = df['high'].iloc[-3:].max()
                price_near_head = abs(current_price - inverse.head_low_price) / current_price < 0.01
                price_dropping = current_price < recent_high
                entry_valid = price_near_head and price_dropping
            
            if self.entry_on_breakout and not entry_valid:
                entry_valid = current_price < inverse.neck_high_price * 0.999
            
            if entry_valid:
                # Stop loss above double top
                stop_loss = max(inverse.first_low_price, inverse.second_low_price) * 1.002
                
                # Take profit using extensions
                highest_high = max(inverse.first_low_price, inverse.second_low_price)
                fib_base = highest_high - inverse.back_high_price
                take_profit = inverse.back_high_price - (fib_base * self.take_profit_fib_levels[0])
                
                position_size = self.calculate_position_size(
                    account_balance, current_price, stop_loss
                )
                
                return TradeSignal(
                    signal=Signal.SHORT,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=position_size,
                    reason=f"INVERSE BULLDOG SHORT: Pattern strength {inverse.pattern_strength:.0%}"
                )
        
        logger.info(f"No Bulldog pattern detected. Price: {current_price:.2f}")
        return None


# Alias for compatibility
Strategy = BulldogStrategy
