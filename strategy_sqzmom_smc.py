"""
Squeeze Momentum Strategy
==========================
Pure TTM Squeeze: trades when volatility compresses then expands.
3 indicators only: Bollinger Bands, Keltner Channels, Momentum + ATR for sizing.
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


class SqzMomSmcStrategy:
    """
    Pure squeeze momentum strategy.
    Trades when squeeze fires (BB crosses out of KC) with momentum confirmation.
    """

    def __init__(
        self,
        # Squeeze params
        bb_length: int = 20,
        bb_mult: float = 2.0,
        kc_length: int = 20,
        kc_mult: float = 1.5,
        mom_length: int = 12,
        # Risk
        atr_period: int = 14,
        sl_atr_mult: float = 1.5,
        tp_atr_mult: float = 2.5,
        risk_per_trade: float = 0.02,
        leverage: int = 4,
        # Ignored (kept for config compatibility)
        **kwargs,
    ):
        self.bb_length = bb_length
        self.bb_mult = bb_mult
        self.kc_length = kc_length
        self.kc_mult = kc_mult
        self.mom_length = mom_length
        self.atr_period = atr_period
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage

    def calculate_squeeze(self, df: pd.DataFrame) -> pd.DataFrame:
        """TTM Squeeze: BB vs KC + linear regression momentum."""
        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # True Range / ATR
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.kc_length).mean()

        # Bollinger Bands
        bb_mid = close.rolling(self.bb_length).mean()
        bb_std = close.rolling(self.bb_length).std()
        bb_upper = bb_mid + self.bb_mult * bb_std
        bb_lower = bb_mid - self.bb_mult * bb_std

        # Keltner Channels
        kc_mid = close.ewm(span=self.kc_length, adjust=False).mean()
        kc_upper = kc_mid + self.kc_mult * atr
        kc_lower = kc_mid - self.kc_mult * atr

        # Squeeze state
        df["sqz_on"] = (bb_lower > kc_lower) & (bb_upper < kc_upper)

        # Momentum via linear regression
        midline = (kc_mid + bb_mid) / 2
        delta = close - midline

        mom_values = np.full(len(df), np.nan)
        x = np.arange(self.mom_length)
        for i in range(self.mom_length - 1, len(df)):
            y = delta.iloc[i - self.mom_length + 1 : i + 1].values
            if np.any(np.isnan(y)):
                continue
            coeffs = np.polyfit(x, y, 1)
            mom_values[i] = np.polyval(coeffs, self.mom_length - 1)

        df["sqz_mom"] = mom_values
        df["sqz_mom_prev"] = pd.Series(mom_values).shift(1).values
        df["atr"] = tr.rolling(self.atr_period).mean()

        return df

    def calculate_position_size(
        self, balance: float, entry: float, stop_loss: float
    ) -> float:
        risk_amount = balance * self.risk_per_trade
        price_risk = abs(entry - stop_loss)
        if price_risk < 1e-10:
            return 0
        # Returns BTC quantity — PnL = size * price_change (no leverage mult needed)
        return risk_amount / price_risk

    def analyze(
        self, df: pd.DataFrame, account_balance: float
    ) -> Optional[TradeSignal]:
        if len(df) < 50:
            return None

        df = self.calculate_squeeze(df)

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        price = curr["close"]
        atr = curr["atr"]
        mom = curr["sqz_mom"]
        mom_prev = curr["sqz_mom_prev"]

        if pd.isna(atr) or atr == 0 or pd.isna(mom) or pd.isna(mom_prev) or np.isinf(atr) or np.isinf(mom) or np.isinf(mom_prev):
            return None

        sqz_on = bool(curr["sqz_on"])
        sqz_was_on = bool(prev["sqz_on"])
        squeeze_fired = sqz_was_on and not sqz_on

        # Log state
        logger.info(f"[*] SQUEEZE: ON={sqz_on} | Fired={squeeze_fired} | Mom={mom:.2f} | Prev={mom_prev:.2f}")

        # --- Decision logic ---
        # Primary: squeeze fires + momentum direction
        # Secondary: momentum accelerating (no squeeze fire needed, but stronger signal)
        direction = None

        if squeeze_fired:
            # Squeeze just released — high conviction
            if mom > 0:
                direction = "LONG"
                reason = f"SQUEEZE FIRE LONG (mom={mom:.1f})"
            elif mom < 0:
                direction = "SHORT"
                reason = f"SQUEEZE FIRE SHORT (mom={mom:.1f})"
        elif not sqz_on:
            # Squeeze already off — momentum continuation
            # Need: momentum strong + accelerating + crossed zero recently
            mom_crossed_up = mom > 0 and mom_prev <= 0
            mom_crossed_down = mom < 0 and mom_prev >= 0
            mom_accelerating_up = mom > 0 and mom > mom_prev
            mom_accelerating_down = mom < 0 and mom < mom_prev

            if mom_crossed_up and mom_accelerating_up:
                direction = "LONG"
                reason = f"MOM CROSS LONG (mom={mom:.1f})"
            elif mom_crossed_down and mom_accelerating_down:
                direction = "SHORT"
                reason = f"MOM CROSS SHORT (mom={mom:.1f})"

        if direction is None:
            logger.info("    No signal")
            return None

        # Build trade
        if direction == "LONG":
            stop_loss = price - (atr * self.sl_atr_mult)
            take_profit = price + (atr * self.tp_atr_mult)
            signal = Signal.LONG
        else:
            stop_loss = price + (atr * self.sl_atr_mult)
            take_profit = price - (atr * self.tp_atr_mult)
            signal = Signal.SHORT

        position_size = self.calculate_position_size(
            account_balance, price, stop_loss
        )

        logger.info(f"[{'+'if direction=='LONG' else '-'}] {reason}")

        return TradeSignal(
            signal=signal,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            reason=reason,
        )


Strategy = SqzMomSmcStrategy
