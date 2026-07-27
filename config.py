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
    STRATEGY: str = "sqzmom_smc"  # Options: "bulldog", "ema_rsi", "donchian", "vwap", "orderflow", "multi", "probability", "sqzmom_smc"
    
    # ===================
    # API Configuration
    # ===================
    API_KEY: str = os.getenv("POLONIEX_API_KEY", "")
    API_SECRET: str = os.getenv("POLONIEX_API_SECRET", "")
    
    # ===================
    # Trading Parameters
    # ===================
    SYMBOL: str = "BTC/USDT:USDT"
    # 5m per user decision 2026-06-11 (rolled back from 4h). NOTE: walk-forward
    # backtest (backtest_entry_filters.py) shows 5m is negative-expectancy after
    # fees; the validated profitable config is TIMEFRAME="4h" + SQZ mults 2.0/4.0
    # + HTF_CONFLUENCE_ENABLED=True. Flip those four values to restore it.
    TIMEFRAME: str = "5m"
    LEVERAGE: int = 10
    MARGIN_MODE: str = "cross"
    
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
    RISK_PER_TRADE: float = 0.02  # 2% of account per trade (reduced from 3%)
    STOP_LOSS_PCT: float = 0.015  # 1.5% stop (tighter for scalping)
    TAKE_PROFIT_PCT: float = 0.01 # 1% profit (quick exits)
    MAX_DAILY_LOSS: float = 0.08  # 8% max daily loss (tighter protection)
    
    # ===================
    # Squeeze Momentum Parameters (AGGRESSIVE)
    # ===================
    SQZ_BB_LENGTH: int = 16       # Shorter BB = more responsive squeezes
    SQZ_BB_MULT: float = 1.8      # Tighter bands = squeeze fires more often
    SQZ_KC_LENGTH: int = 16       # Shorter KC = faster detection
    SQZ_KC_MULT: float = 1.0      # Tighter KC = many more squeezes detected
    SQZ_MOM_LENGTH: int = 8        # Shorter momentum = faster signals
    SQZ_SL_ATR_MULT: float = 1.2   # 4h-validated value: 2.0
    SQZ_TP_ATR_MULT: float = 3.6   # 4h-validated value: 4.0
    HTF_CONFLUENCE_ENABLED: bool = False  # 4h-validated value: True (daily EMA20 filter)

    # ===================
    # Regime Filter (entry gating)
    # ===================
    # Data showed counter-trend entries and ranging-market entries are the bot's
    # losing buckets. These gates block them at the orchestration layer using the
    # MarketScout snapshot. Toggle off to restore prior (unfiltered) behavior.
    REGIME_FILTER_ENABLED: bool = True
    BLOCK_COUNTER_TREND: bool = True       # block LONG in BEAR / SHORT in BULL
    BLOCK_RANGING_REGIME: bool = True      # block entries when regime == RANGING
    COUNTER_TREND_MIN_STRENGTH: float = 0.0  # only block counter-trend if trend_strength >= this

    # ===================
    # Logging
    # ===================
    LOG_FILE: str = "trading_bot.log"


config = Config()
