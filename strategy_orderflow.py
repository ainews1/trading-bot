"""
Order Flow Enhanced Strategy
============================
Combines technical signals with order flow data from Binance
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import logging

from external_data import BinanceFetcher, OrderFlowData, get_order_flow_signal
from data_providers import HyperliquidAPI, LiquidationTracker, MarketDataAggregator

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


class OrderFlowStrategy:
    """
    EMA Crossover + Order Flow Filter
    
    Rules:
    - Base: EMA 8/24 crossover
    - Filter: Only take trades aligned with order flow sentiment
    - Funding rate extreme = fade the crowd
    - L/S ratio extreme = contrarian signal
    """
    
    def __init__(
        self,
        fast_ema: int = 8,
        slow_ema: int = 24,
        atr_period: int = 14,
        sl_atr_mult: float = 1.5,
        tp_atr_mult: float = 2.0,
        risk_per_trade: float = 0.01,
        leverage: int = 5,
        use_orderflow_filter: bool = True,
        min_sentiment_strength: float = 10.0,  # Min sentiment score to trade
    ):
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.atr_period = atr_period
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage
        self.use_orderflow_filter = use_orderflow_filter
        self.min_sentiment_strength = min_sentiment_strength
        
        self.data_fetcher = BinanceFetcher()
        self.hyperliquid = HyperliquidAPI()
        self.liq_tracker = LiquidationTracker()
        self.last_orderflow: Optional[OrderFlowData] = None
    
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
    
    def get_orderflow_bias(self) -> tuple:
        """
        Get order flow bias from multiple sources
        Returns: (bias: str, confidence: float, data: OrderFlowData)
        """
        data = self.data_fetcher.get_aggregated_data()
        self.last_orderflow = data
        
        bias = "NEUTRAL"
        confidence = 0.0
        signals = []
        
        # Binance sentiment
        if data.sentiment_score is not None:
            if data.sentiment_score > self.min_sentiment_strength:
                signals.append(("BULLISH", 0.3))
            elif data.sentiment_score < -self.min_sentiment_strength:
                signals.append(("BEARISH", 0.3))
        
        # Hyperliquid order book imbalance
        try:
            ob = self.hyperliquid.get_orderbook_imbalance("BTC")
            if ob["imbalance_ratio"] > 2.0:  # Strong bid support
                signals.append(("BULLISH", 0.4))
                logger.info(f"📊 Hyperliquid: Bid imbalance {ob['imbalance_ratio']:.1f}x - BULLISH")
            elif ob["imbalance_ratio"] < 0.5:  # Strong ask pressure
                signals.append(("BEARISH", 0.4))
                logger.info(f"📊 Hyperliquid: Ask imbalance {1/ob['imbalance_ratio']:.1f}x - BEARISH")
            
            # Log walls
            if ob.get("bid_wall"):
                logger.info(f"   Bid Wall: {ob['bid_wall']['size']:.1f} BTC @ ${ob['bid_wall']['price']:,.0f}")
            if ob.get("ask_wall"):
                logger.info(f"   Ask Wall: {ob['ask_wall']['size']:.1f} BTC @ ${ob['ask_wall']['price']:,.0f}")
        except Exception as e:
            logger.warning(f"Hyperliquid fetch error: {e}")
        
        # Funding rate signal
        if data.funding_rate is not None:
            if data.funding_rate > 0.001:  # 0.1% = crowded long
                signals.append(("BEARISH", 0.25))
                logger.info(f"⚠️ High funding {data.funding_rate*100:.3f}% - contrarian BEARISH")
            elif data.funding_rate < -0.001:  # -0.1% = crowded short
                signals.append(("BULLISH", 0.25))
                logger.info(f"⚠️ Negative funding {data.funding_rate*100:.3f}% - contrarian BULLISH")
        
        # L/S ratio signal
        if data.long_short_ratio is not None:
            if data.long_short_ratio > 2.5:  # Very crowded long
                signals.append(("BEARISH", 0.2))
                logger.info(f"⚠️ Crowded long L/S={data.long_short_ratio:.2f}")
            elif data.long_short_ratio < 0.5:  # Very crowded short
                signals.append(("BULLISH", 0.2))
                logger.info(f"⚠️ Crowded short L/S={data.long_short_ratio:.2f}")
        
        # Aggregate signals
        if signals:
            bull_score = sum(conf for b, conf in signals if b == "BULLISH")
            bear_score = sum(conf for b, conf in signals if b == "BEARISH")
            
            if bull_score > bear_score + 0.15:
                bias = "BULLISH"
                confidence = min(bull_score, 1.0)
            elif bear_score > bull_score + 0.15:
                bias = "BEARISH"
                confidence = min(bear_score, 1.0)
            else:
                confidence = abs(bull_score - bear_score)
        
        return bias, confidence, data
    
    def analyze(self, df: pd.DataFrame, account_balance: float) -> Optional[TradeSignal]:
        if len(df) < 50:
            return None
        
        df = df.copy()
        df['ema_fast'] = df['close'].ewm(span=self.fast_ema, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_ema, adjust=False).mean()
        df['atr'] = self.calculate_atr(df)
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        price = current['close']
        atr = current['atr']
        
        if pd.isna(atr):
            return None
        
        # Check for EMA crossover
        cross_up = prev['ema_fast'] <= prev['ema_slow'] and current['ema_fast'] > current['ema_slow']
        cross_down = prev['ema_fast'] >= prev['ema_slow'] and current['ema_fast'] < current['ema_slow']
        
        if not cross_up and not cross_down:
            return None
        
        # Get order flow bias
        of_bias, of_confidence, of_data = self.get_orderflow_bias()
        
        logger.info(f"📊 Order Flow: {of_bias} (confidence: {of_confidence:.1%})")
        logger.info(f"   Funding: {of_data.funding_rate*100:.4f}%" if of_data.funding_rate else "   Funding: N/A")
        logger.info(f"   L/S Ratio: {of_data.long_short_ratio:.2f}" if of_data.long_short_ratio else "   L/S: N/A")
        logger.info(f"   Sentiment: {of_data.sentiment_score:+.1f}" if of_data.sentiment_score else "   Sentiment: N/A")
        
        # LONG signal
        if cross_up:
            # Check order flow filter
            if self.use_orderflow_filter:
                if of_bias == "BEARISH":
                    logger.info("❌ LONG rejected: Order flow bearish")
                    return None
                if of_data.long_short_ratio and of_data.long_short_ratio > 2.5:
                    logger.info("❌ LONG rejected: Crowded long position")
                    return None
            
            stop_loss = price - (atr * self.sl_atr_mult)
            take_profit = price + (atr * self.tp_atr_mult)
            
            # Adjust TP based on confidence
            if of_bias == "BULLISH" and of_confidence > 0.5:
                take_profit = price + (atr * self.tp_atr_mult * 1.25)  # Extend TP
                logger.info("✨ Extended TP due to bullish order flow")
            
            position_size = self.calculate_position_size(account_balance, price, stop_loss)
            
            return TradeSignal(
                signal=Signal.LONG,
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reason=f"ORDERFLOW LONG: EMA cross + {of_bias} flow (L/S: {of_data.long_short_ratio:.2f})" if of_data.long_short_ratio else "ORDERFLOW LONG: EMA cross"
            )
        
        # SHORT signal
        if cross_down:
            if self.use_orderflow_filter:
                if of_bias == "BULLISH":
                    logger.info("❌ SHORT rejected: Order flow bullish")
                    return None
                if of_data.long_short_ratio and of_data.long_short_ratio < 0.5:
                    logger.info("❌ SHORT rejected: Crowded short position")
                    return None
            
            stop_loss = price + (atr * self.sl_atr_mult)
            take_profit = price - (atr * self.tp_atr_mult)
            
            if of_bias == "BEARISH" and of_confidence > 0.5:
                take_profit = price - (atr * self.tp_atr_mult * 1.25)
                logger.info("✨ Extended TP due to bearish order flow")
            
            position_size = self.calculate_position_size(account_balance, price, stop_loss)
            
            return TradeSignal(
                signal=Signal.SHORT,
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reason=f"ORDERFLOW SHORT: EMA cross + {of_bias} flow (L/S: {of_data.long_short_ratio:.2f})" if of_data.long_short_ratio else "ORDERFLOW SHORT: EMA cross"
            )
        
        return None


Strategy = OrderFlowStrategy
