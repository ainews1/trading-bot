"""
Bitcoin Market Scout
====================
Advanced market analysis module for Bitcoin trading
Provides comprehensive market intelligence including:
- Volume Profile Analysis
- Volatility Metrics
- Trend Strength Indicators
- Support/Resistance Levels
- Market Regime Detection
- Order Flow Insights
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarketSnapshot:
    """Comprehensive market analysis snapshot"""
    current_price: float
    trend_direction: str
    trend_strength: float
    volatility_regime: str
    volume_profile: Dict
    support_levels: List[float]
    resistance_levels: List[float]
    market_regime: str
    momentum_score: float
    liquidity_zones: List[Dict]
    risk_score: float


class MarketScout:
    """
    Advanced Bitcoin market analysis and scouting system
    """
    
    def __init__(self, volume_profile_bins: int = 50):
        self.volume_profile_bins = volume_profile_bins
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range for volatility measurement"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def calculate_volume_profile(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Volume Profile - distribution of volume across price levels
        Returns VPOC, Value Area High/Low, and volume distribution
        """
        if len(df) < 20:
            return {}
        
        price_min = df['low'].min()
        price_max = df['high'].max()
        price_range = price_max - price_min
        
        if price_range == 0:
            return {}
        
        bins = np.linspace(price_min, price_max, self.volume_profile_bins)
        volume_distribution = np.zeros(len(bins) - 1)
        
        for idx, row in df.iterrows():
            low_price = row['low']
            high_price = row['high']
            volume = row['volume']
            
            if volume <= 0:
                continue
            
            low_bin = np.digitize([low_price], bins)[0] - 1
            high_bin = np.digitize([high_price], bins)[0] - 1
            
            low_bin = max(0, min(low_bin, len(volume_distribution) - 1))
            high_bin = max(0, min(high_bin, len(volume_distribution) - 1))
            
            if low_bin == high_bin:
                volume_distribution[low_bin] += volume
            else:
                price_range_in_candle = high_price - low_price
                if price_range_in_candle > 0:
                    for bin_idx in range(low_bin, high_bin + 1):
                        bin_low = bins[bin_idx] if bin_idx < len(bins) else bins[-1]
                        bin_high = bins[bin_idx + 1] if bin_idx + 1 < len(bins) else bins[-1]
                        
                        overlap_low = max(low_price, bin_low)
                        overlap_high = min(high_price, bin_high)
                        overlap = max(0, overlap_high - overlap_low)
                        
                        volume_contribution = volume * (overlap / price_range_in_candle)
                        if bin_idx < len(volume_distribution):
                            volume_distribution[bin_idx] += volume_contribution
        
        vpoc_idx = np.argmax(volume_distribution)
        vpoc_price = (bins[vpoc_idx] + bins[vpoc_idx + 1]) / 2
        
        total_volume = volume_distribution.sum()
        cumulative_volume = 0
        value_area_high_idx = vpoc_idx
        value_area_low_idx = vpoc_idx
        
        target_volume = total_volume * 0.70
        
        while cumulative_volume < target_volume:
            if value_area_low_idx > 0 and value_area_high_idx < len(volume_distribution) - 1:
                low_vol = volume_distribution[value_area_low_idx - 1]
                high_vol = volume_distribution[value_area_high_idx + 1]
                
                if low_vol > high_vol:
                    cumulative_volume += low_vol
                    value_area_low_idx -= 1
                else:
                    cumulative_volume += high_vol
                    value_area_high_idx += 1
            elif value_area_low_idx > 0:
                cumulative_volume += volume_distribution[value_area_low_idx - 1]
                value_area_low_idx -= 1
            elif value_area_high_idx < len(volume_distribution) - 1:
                cumulative_volume += volume_distribution[value_area_high_idx + 1]
                value_area_high_idx += 1
            else:
                break
        
        vah_price = bins[value_area_high_idx + 1] if value_area_high_idx + 1 < len(bins) else bins[-1]
        val_price = bins[value_area_low_idx] if value_area_low_idx < len(bins) else bins[0]
        
        return {
            'vpoc': float(vpoc_price),
            'value_area_high': float(vah_price),
            'value_area_low': float(val_price),
            'total_volume': float(total_volume),
            'distribution': volume_distribution.tolist(),
            'price_bins': bins.tolist()
        }
    
    def find_support_resistance(self, df: pd.DataFrame, lookback: int = 50) -> Tuple[List[float], List[float]]:
        """
        Identify support and resistance levels using swing highs/lows
        """
        if len(df) < lookback:
            lookback = len(df)
        
        recent_df = df.iloc[-lookback:].copy()
        
        swing_lookback = 5
        support_levels = []
        resistance_levels = []
        
        for i in range(swing_lookback, len(recent_df) - swing_lookback):
            low_price = recent_df['low'].iloc[i]
            high_price = recent_df['high'].iloc[i]
            
            window_lows = recent_df['low'].iloc[i-swing_lookback:i+swing_lookback+1]
            window_highs = recent_df['high'].iloc[i-swing_lookback:i+swing_lookback+1]
            
            if low_price == window_lows.min():
                support_levels.append(float(low_price))
            
            if high_price == window_highs.max():
                resistance_levels.append(float(high_price))
        
        support_levels = sorted(set(support_levels), reverse=True)[:5]
        resistance_levels = sorted(set(resistance_levels))[:5]
        
        return support_levels, resistance_levels
    
    def detect_trend(self, df: pd.DataFrame, ema_fast: int = 9, ema_slow: int = 21) -> Tuple[str, float]:
        """
        Detect trend direction and strength using EMA crossovers
        Returns: ('BULL', 'BEAR', 'NEUTRAL'), strength (0-1)
        """
        if len(df) < ema_slow:
            return 'NEUTRAL', 0.0
        
        close = df['close']
        ema_fast_series = close.ewm(span=ema_fast, adjust=False).mean()
        ema_slow_series = close.ewm(span=ema_slow, adjust=False).mean()
        
        current_price = close.iloc[-1]
        fast_ema = ema_fast_series.iloc[-1]
        slow_ema = ema_slow_series.iloc[-1]
        
        if fast_ema > slow_ema and current_price > fast_ema:
            trend = 'BULL'
            strength = min(1.0, (fast_ema - slow_ema) / slow_ema * 100)
        elif fast_ema < slow_ema and current_price < fast_ema:
            trend = 'BEAR'
            strength = min(1.0, (slow_ema - fast_ema) / fast_ema * 100)
        else:
            trend = 'NEUTRAL'
            strength = 0.0
        
        return trend, float(strength)
    
    def calculate_momentum(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate momentum score using RSI and price change
        Returns score from -1 (strong bearish) to 1 (strong bullish)
        """
        if len(df) < period + 1:
            return 0.0
        
        close = df['close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        price_change = (close.iloc[-1] - close.iloc[-period]) / close.iloc[-period]
        
        momentum = ((current_rsi - 50) / 50) * 0.6 + price_change * 0.4
        
        return float(np.clip(momentum, -1.0, 1.0))
    
    def detect_volatility_regime(self, df: pd.DataFrame, period: int = 20) -> str:
        """
        Detect current volatility regime: LOW, NORMAL, HIGH, EXTREME
        """
        if len(df) < period:
            return 'NORMAL'
        
        atr = self.calculate_atr(df, period=14)
        current_atr = atr.iloc[-1]
        avg_atr = atr.iloc[-period:].mean()
        
        if avg_atr == 0:
            return 'NORMAL'
        
        volatility_ratio = current_atr / avg_atr
        
        if volatility_ratio < 0.7:
            return 'LOW'
        elif volatility_ratio < 1.3:
            return 'NORMAL'
        elif volatility_ratio < 2.0:
            return 'HIGH'
        else:
            return 'EXTREME'
    
    def detect_market_regime(self, df: pd.DataFrame) -> str:
        """
        Detect overall market regime: TRENDING, RANGING, BREAKOUT, REVERSAL
        """
        if len(df) < 50:
            return 'RANGING'
        
        close = df['close']
        high = df['high']
        low = df['low']
        
        ema_20 = close.ewm(span=20, adjust=False).mean()
        ema_50 = close.ewm(span=50, adjust=False).mean()
        
        price_range_20 = (high.iloc[-20:].max() - low.iloc[-20:].min()) / close.iloc[-1]
        price_range_50 = (high.iloc[-50:].max() - low.iloc[-50:].min()) / close.iloc[-1]
        
        trend, strength = self.detect_trend(df)
        
        if strength > 0.3:
            return 'TRENDING'
        elif price_range_20 < price_range_50 * 0.5:
            return 'RANGING'
        elif abs(close.iloc[-1] - ema_20.iloc[-1]) / close.iloc[-1] > 0.02:
            return 'BREAKOUT'
        else:
            return 'REVERSAL'
    
    def identify_liquidity_zones(self, df: pd.DataFrame, volume_profile: Dict) -> List[Dict]:
        """
        Identify high and low liquidity zones based on volume profile
        """
        zones = []
        
        if not volume_profile or 'distribution' not in volume_profile:
            return zones
        
        distribution = np.array(volume_profile['distribution'])
        price_bins = volume_profile.get('price_bins', [])
        
        if len(distribution) == 0 or len(price_bins) < 2:
            return zones
        
        mean_volume = distribution.mean()
        std_volume = distribution.std()
        
        high_volume_threshold = mean_volume + std_volume
        low_volume_threshold = mean_volume - std_volume
        
        for i in range(len(distribution)):
            volume = distribution[i]
            price_low = price_bins[i] if i < len(price_bins) else price_bins[-1]
            price_high = price_bins[i + 1] if i + 1 < len(price_bins) else price_bins[-1]
            price_center = (price_low + price_high) / 2
            
            if volume > high_volume_threshold:
                zones.append({
                    'type': 'HIGH_LIQUIDITY',
                    'price': float(price_center),
                    'price_low': float(price_low),
                    'price_high': float(price_high),
                    'volume': float(volume),
                    'significance': 'HIGH'
                })
            elif volume < low_volume_threshold and volume > 0:
                zones.append({
                    'type': 'LOW_LIQUIDITY',
                    'price': float(price_center),
                    'price_low': float(price_low),
                    'price_high': float(price_high),
                    'volume': float(volume),
                    'significance': 'MEDIUM'
                })
        
        return sorted(zones, key=lambda x: x['volume'], reverse=True)[:10]
    
    def calculate_risk_score(self, df: pd.DataFrame, volatility_regime: str) -> float:
        """
        Calculate overall market risk score (0-1, higher = more risky)
        """
        risk_factors = []
        
        atr = self.calculate_atr(df, period=14)
        current_atr = atr.iloc[-1]
        avg_atr = atr.iloc[-20:].mean() if len(df) >= 20 else current_atr
        
        if avg_atr > 0:
            volatility_risk = min(1.0, current_atr / avg_atr / 2.0)
            risk_factors.append(volatility_risk)
        
        volume = df['volume']
        avg_volume = volume.iloc[-20:].mean() if len(df) >= 20 else volume.mean()
        current_volume = volume.iloc[-1]
        
        if avg_volume > 0:
            volume_risk = 1.0 - min(1.0, current_volume / avg_volume)
            risk_factors.append(volume_risk)
        
        close = df['close']
        price_change = abs(close.iloc[-1] - close.iloc[-10]) / close.iloc[-10] if len(df) >= 10 else 0
        price_risk = min(1.0, price_change * 10)
        risk_factors.append(price_risk)
        
        if volatility_regime == 'EXTREME':
            risk_factors.append(0.9)
        elif volatility_regime == 'HIGH':
            risk_factors.append(0.6)
        elif volatility_regime == 'LOW':
            risk_factors.append(0.3)
        else:
            risk_factors.append(0.5)
        
        return float(np.mean(risk_factors)) if risk_factors else 0.5
    
    def scout_market(self, df: pd.DataFrame) -> MarketSnapshot:
        """
        Comprehensive market analysis - main scouting function
        """
        if len(df) < 20:
            logger.warning("Insufficient data for market scouting")
            return None
        
        current_price = float(df['close'].iloc[-1])
        
        volume_profile = self.calculate_volume_profile(df)
        trend, trend_strength = self.detect_trend(df)
        volatility_regime = self.detect_volatility_regime(df)
        market_regime = self.detect_market_regime(df)
        momentum_score = self.calculate_momentum(df)
        support_levels, resistance_levels = self.find_support_resistance(df)
        liquidity_zones = self.identify_liquidity_zones(df, volume_profile)
        risk_score = self.calculate_risk_score(df, volatility_regime)
        
        return MarketSnapshot(
            current_price=current_price,
            trend_direction=trend,
            trend_strength=trend_strength,
            volatility_regime=volatility_regime,
            volume_profile=volume_profile,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            market_regime=market_regime,
            momentum_score=momentum_score,
            liquidity_zones=liquidity_zones,
            risk_score=risk_score
        )
    
    def format_snapshot_report(self, snapshot: MarketSnapshot) -> str:
        """Format market snapshot as readable report"""
        if snapshot is None:
            return "No market data available"
        
        report = []
        report.append("=" * 60)
        report.append("BITCOIN MARKET SCOUT REPORT")
        report.append("=" * 60)
        report.append(f"Current Price: ${snapshot.current_price:,.2f}")
        report.append(f"Trend: {snapshot.trend_direction} (Strength: {snapshot.trend_strength:.1%})")
        report.append(f"Market Regime: {snapshot.market_regime}")
        report.append(f"Volatility: {snapshot.volatility_regime}")
        report.append(f"Momentum Score: {snapshot.momentum_score:+.2f}")
        report.append(f"Risk Score: {snapshot.risk_score:.1%}")
        
        if snapshot.volume_profile:
            vpoc = snapshot.volume_profile.get('vpoc', 0)
            vah = snapshot.volume_profile.get('value_area_high', 0)
            val = snapshot.volume_profile.get('value_area_low', 0)
            report.append("")
            report.append("Volume Profile:")
            report.append(f"  VPOC: ${vpoc:,.2f}")
            report.append(f"  Value Area High: ${vah:,.2f}")
            report.append(f"  Value Area Low: ${val:,.2f}")
        
        if snapshot.support_levels:
            report.append("")
            report.append("Support Levels:")
            for level in snapshot.support_levels[:3]:
                report.append(f"  ${level:,.2f}")
        
        if snapshot.resistance_levels:
            report.append("")
            report.append("Resistance Levels:")
            for level in snapshot.resistance_levels[:3]:
                report.append(f"  ${level:,.2f}")
        
        if snapshot.liquidity_zones:
            report.append("")
            report.append("Key Liquidity Zones:")
            for zone in snapshot.liquidity_zones[:3]:
                report.append(f"  {zone['type']}: ${zone['price']:,.2f} (Vol: {zone['volume']:.0f})")
        
        report.append("=" * 60)
        
        return "\n".join(report)
