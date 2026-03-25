"""
RSI(3) Mean Reversion Scalper v5
=================================
Based on Connors RSI research adapted for 5-min BTC scalping.

RSI(3) is extremely sensitive — it captures 3-candle exhaustion moves.
When RSI(3) hits extreme levels in a trending market, price almost
always snaps back. We capture that snap-back.

Entry (LONG):
  1. RSI(3) < 10 (extreme 3-candle selloff)
  2. Price above EMA(50) (uptrend — pullback, not collapse)
  3. EMA(20) > EMA(50) (trend confirmed)

Entry (SHORT):
  1. RSI(3) > 90 (extreme 3-candle rally)
  2. Price below EMA(50) (downtrend — bounce, not breakout)
  3. EMA(20) < EMA(50) (downtrend confirmed)

Exit:
  Dynamic: RSI(3) crosses above/below 50 (mean reversion complete)
  Stop: 2.0x ATR (wide — give room for the snap-back)
  Max hold: 20 candles (100 min — if no reversion, cut loss)
"""

import numpy as np
import pandas as pd
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


class ScalperStrategy:

    def __init__(
        self,
        # RSI
        rsi_fast_period: int = 3,         # The 3-period RSI for entries
        rsi_entry_oversold: float = 10,    # Extreme oversold
        rsi_entry_overbought: float = 90,  # Extreme overbought
        rsi_exit_level: float = 50,        # Exit when RSI crosses 50
        # Trend filter EMAs
        ema_fast: int = 20,
        ema_slow: int = 50,
        # ATR for stops
        atr_period: int = 14,
        atr_sl_multiplier: float = 2.0,   # Wide stop
        atr_volatility_cap: float = 2.5,
        atr_avg_period: int = 50,
        # Trade management
        max_hold_candles: int = 20,
        min_candles_between: int = 6,
        # Risk
        risk_per_trade: float = 0.015,
        leverage: int = 5,
    ):
        self.rsi_fast_period = rsi_fast_period
        self.rsi_entry_oversold = rsi_entry_oversold
        self.rsi_entry_overbought = rsi_entry_overbought
        self.rsi_exit_level = rsi_exit_level
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.atr_period = atr_period
        self.atr_sl_multiplier = atr_sl_multiplier
        self.atr_volatility_cap = atr_volatility_cap
        self.atr_avg_period = atr_avg_period
        self.max_hold_candles = max_hold_candles
        self.min_candles_between = min_candles_between
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df['close']
        high = df['high']
        low = df['low']

        # RSI(3) — Wilder's smoothing
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss_s = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0 / self.rsi_fast_period, min_periods=self.rsi_fast_period, adjust=False).mean()
        avg_loss = loss_s.ewm(alpha=1.0 / self.rsi_fast_period, min_periods=self.rsi_fast_period, adjust=False).mean()
        rs = avg_gain / avg_loss
        df['rsi3'] = 100.0 - (100.0 / (1.0 + rs))

        # Trend EMAs
        df['ema_fast'] = close.ewm(span=self.ema_fast, adjust=False).mean()
        df['ema_slow'] = close.ewm(span=self.ema_slow, adjust=False).mean()

        # ATR
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(self.atr_period).mean()
        df['atr_avg'] = df['atr'].rolling(self.atr_avg_period).mean()

        return df

    def calculate_position_size(self, balance: float, entry: float, stop_loss: float) -> float:
        risk_amount = balance * self.risk_per_trade
        price_risk = abs(entry - stop_loss)
        if price_risk == 0:
            return 0.0
        return risk_amount / price_risk

    def check_signal(self, df: pd.DataFrame, idx: int) -> Signal:
        row = df.iloc[idx]

        if pd.isna(row['rsi3']) or pd.isna(row['ema_slow']) or pd.isna(row['atr_avg']):
            return Signal.NONE

        price = row['close']
        rsi3 = row['rsi3']
        atr = row['atr']
        atr_avg = row['atr_avg']

        # Volatility filter
        if atr_avg > 0 and atr > atr_avg * self.atr_volatility_cap:
            return Signal.NONE

        uptrend = row['ema_fast'] > row['ema_slow'] and price > row['ema_slow']
        downtrend = row['ema_fast'] < row['ema_slow'] and price < row['ema_slow']

        # LONG: RSI(3) extreme oversold in uptrend
        if rsi3 < self.rsi_entry_oversold and uptrend:
            return Signal.LONG

        # SHORT: RSI(3) extreme overbought in downtrend
        if rsi3 > self.rsi_entry_overbought and downtrend:
            return Signal.SHORT

        return Signal.NONE

    def get_levels(self, df: pd.DataFrame, idx: int, signal: Signal):
        """Returns entry and SL. TP is dynamic (RSI exit), so TP here is max target."""
        row = df.iloc[idx]
        entry = row['close']
        atr = row['atr']

        if signal == Signal.LONG:
            sl = entry - (atr * self.atr_sl_multiplier)
            tp = entry + (atr * 3.0)  # Max target (usually exits earlier via RSI)
        else:
            sl = entry + (atr * self.atr_sl_multiplier)
            tp = entry - (atr * 3.0)

        return entry, sl, tp

    def analyze(self, df: pd.DataFrame, account_balance: float) -> Optional[TradeSignal]:
        if len(df) < self.ema_slow + 10:
            return None
        df = self.compute_indicators(df)
        idx = len(df) - 1
        signal = self.check_signal(df, idx)
        if signal == Signal.NONE:
            return None
        entry, sl, tp = self.get_levels(df, idx, signal)
        position_size = self.calculate_position_size(account_balance, entry, sl)
        row = df.iloc[idx]
        return TradeSignal(
            signal=signal,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            position_size=position_size,
            reason=(
                f"{'LONG' if signal == Signal.LONG else 'SHORT'}: "
                f"RSI(3)={row['rsi3']:.1f}, trend={'UP' if signal == Signal.LONG else 'DOWN'}"
            )
        )


Strategy = ScalperStrategy
