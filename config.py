"""
Trading Bot Configuration
=========================
⚠️ WARNING: Set PAPER_TRADING = False ONLY after extensive testing.
Real money will be at risk.
"""

import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # ===================
    # SAFETY FLAG
    # ===================
    PAPER_TRADING: bool = True  # SET TO FALSE ONLY WHEN READY FOR LIVE
    
    # ===================
    # STRATEGY SELECTION
    # ===================
    STRATEGY: str = "bulldog"  # Options: "bulldog", "ema_rsi"
    
    # ===================
    # API Configuration
    # ===================
    API_KEY: str = os.getenv("POLONIEX_API_KEY", "")
    API_SECRET: str = os.getenv("POLONIEX_API_SECRET", "")
    
    # ===================
    # Trading Parameters
    # ===================
    SYMBOL: str = "BTC/USDT:USDT"
    TIMEFRAME: str = "5m"
    LEVERAGE: int = 5
    MARGIN_MODE: str = "isolated"
    
    # ===================
    # BULLDOG Strategy Parameters
    # ===================
    BULLDOG_LOOKBACK: int = 50           # Candles to scan for pattern
    BULLDOG_SWING_LOOKBACK: int = 5      # Candles for swing detection
    BULLDOG_DOUBLE_BOTTOM_TOL: float = 0.005  # 0.5% tolerance
    BULLDOG_MIN_BACK_HEIGHT: float = 0.005    # Min 0.5% for the back
    BULLDOG_MAX_PULLBACK: float = 0.382       # Max 38.2% retracement
    BULLDOG_MIN_PULLBACK: float = 0.10        # Min 10% pullback
    BULLDOG_ENTRY_ON_PULLBACK: bool = True    # Enter on head
    BULLDOG_ENTRY_ON_BREAKOUT: bool = True    # Also enter on breakout
    BULLDOG_TP_FIB_LEVELS: List[float] = field(default_factory=lambda: [0.5, 0.618, 1.0])
    
    # ===================
    # EMA+RSI Strategy Parameters (legacy)
    # ===================
    EMA_PERIOD: int = 30
    RSI_PERIOD: int = 9
    RSI_OVERSOLD: int = 40
    RSI_OVERBOUGHT: int = 60
    VOLUME_MULTIPLIER: float = 0.0
    VOLUME_PERIOD: int = 20
    
    # ===================
    # Risk Management
    # ===================
    RISK_PER_TRADE: float = 0.02  # 2% of account per trade
    STOP_LOSS_PCT: float = 0.03   # 3% stop (fallback)
    TAKE_PROFIT_PCT: float = 0.02 # 2% profit (fallback)
    MAX_DAILY_LOSS: float = 0.06  # 6% max daily loss
    
    # ===================
    # Logging
    # ===================
    LOG_FILE: str = "trading_bot.log"


config = Config()
