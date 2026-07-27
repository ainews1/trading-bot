"""
Squeeze Momentum + SMC Strategy (v4 — AGGRESSIVE)
===================================================
Squeeze momentum with Smart Money Concepts confluence.
Maximized trade frequency with 3 entry types.

Changes from v3:
- ADDED trend continuation entries (no squeeze needed, just strong trend + momentum)
- LOWERED all momentum thresholds by ~50%
- RELAXED EMA filter (uses faster EMA + allows small deviation)
- RELAXED SMC blocking thresholds
- REDUCED cooldown from 6 to 3 candles
- REDUCED consecutive loss trigger from 3 to 4 (more lenient)
- SHORTER indicator periods for faster signals
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import logging

from smc_engine import SmartMoneyEngine

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
    Squeeze Momentum + Smart Money Concepts (v2).

    Entry rules — THREE signal types:

    A) SQUEEZE FIRE (high conviction):
      1. Squeeze fires (BB exits KC)
      2. Momentum confirms direction
      3. EMA trend alignment (loose)
      4. SMC bias not strongly opposing

    B) MOMENTUM ACCELERATION (medium conviction):
      1. Momentum is strong (> 0.08 ATR) and accelerating (increasing for 2 bars)
      2. EMA trend alignment (loose)
      3. SMC bias not strongly opposing
      4. Squeeze is currently ON (building pressure)

    C) TREND CONTINUATION (lower conviction, tightest SL):
      1. Strong momentum (> 0.20 ATR) in direction of EMA trend
      2. Momentum same direction for 3+ bars
      3. No squeeze required
      4. SMC bias not strongly opposing

    Risk management:
      - ATR-based SL/TP
      - Squeeze fire: 1:2.5 R:R
      - Mom accel: 1:2.0 R:R
      - Trend cont: 1:1.8 R:R with tightest SL
      - 2% risk per trade
      - 4-loss consecutive cooldown (3 candles)
    """

    def __init__(
        self,
        # Squeeze params
        bb_length: int = 20,
        bb_mult: float = 2.0,
        kc_length: int = 20,
        kc_mult: float = 1.5,
        mom_length: int = 12,
        # Trend filter
        ema_trend_period: int = 34,  # Faster EMA for quicker trend detection
        ema_tolerance_pct: float = 0.001,  # Allow 0.1% deviation from EMA
        # Risk
        atr_period: int = 10,  # Shorter ATR for more responsive SL/TP
        sl_atr_mult: float = 1.2,
        tp_atr_mult: float = 3.0,
        risk_per_trade: float = 0.02,
        leverage: int = 4,
        # Cooldown
        max_consecutive_losses: int = 4,  # More lenient before cooldown
        cooldown_candles: int = 3,  # Only 3 candles (15 min on 5m) cooldown
        # Volume filter
        volume_confirm_mult: float = 0.8,  # Very relaxed volume filter
        # SMC
        smc_swing_lookback: int = 5,
        min_smc_bias: float = 0.0,  # Minimum SMC bias score (0 = just not opposing)
        # Ignored (kept for config compatibility)
        **kwargs,
    ):
        self.bb_length = bb_length
        self.bb_mult = bb_mult
        self.kc_length = kc_length
        self.kc_mult = kc_mult
        self.mom_length = mom_length
        self.ema_trend_period = ema_trend_period
        self.atr_period = atr_period
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_candles = cooldown_candles
        self.volume_confirm_mult = volume_confirm_mult
        self.min_smc_bias = min_smc_bias
        self.ema_tolerance_pct = ema_tolerance_pct

        # SMC engine for structural analysis
        self.smc = SmartMoneyEngine(swing_lookback=smc_swing_lookback)

        # Cooldown state
        self._consecutive_losses = 0
        self._cooldown_remaining = 0

    def record_trade_result(self, won: bool):
        """Call this after a trade closes to update cooldown tracking."""
        if won:
            self._consecutive_losses = 0
            self._cooldown_remaining = 0
        else:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self.max_consecutive_losses:
                self._cooldown_remaining = self.cooldown_candles
                logger.warning(
                    f"[COOLDOWN] {self._consecutive_losses} consecutive losses — "
                    f"pausing for {self.cooldown_candles} candles"
                )

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

        # EMA trend filter
        df["ema_trend"] = close.ewm(span=self.ema_trend_period, adjust=False).mean()

        # Volume average
        df["vol_avg"] = df["volume"].rolling(20).mean()

        return df

    def calculate_position_size(
        self, balance: float, entry: float, stop_loss: float
    ) -> float:
        risk_amount = balance * self.risk_per_trade
        price_risk = abs(entry - stop_loss)
        if price_risk < 1e-10:
            return 0
        return risk_amount / price_risk

    def analyze(
        self, df: pd.DataFrame, account_balance: float
    ) -> Optional[TradeSignal]:
        if len(df) < max(self.ema_trend_period + 10, 50):
            return None

        df = self.calculate_squeeze(df)

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        price = curr["close"]
        atr = curr["atr"]
        mom = curr["sqz_mom"]
        mom_prev = curr["sqz_mom_prev"]
        ema_trend = curr["ema_trend"]
        vol = curr["volume"]
        vol_avg = curr["vol_avg"]

        # Validate data
        if (pd.isna(atr) or atr == 0 or pd.isna(mom) or pd.isna(mom_prev)
                or np.isinf(atr) or np.isinf(mom) or np.isinf(mom_prev)
                or pd.isna(ema_trend) or pd.isna(vol_avg)):
            return None

        sqz_on = bool(curr["sqz_on"])
        sqz_was_on = bool(prev["sqz_on"])
        squeeze_fired = sqz_was_on and not sqz_on

        # === COOLDOWN CHECK ===
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            logger.info(
                f"[*] SQUEEZE: ON={sqz_on} | Fired={squeeze_fired} | Mom={mom:.2f} "
                f"| COOLDOWN={self._cooldown_remaining} candles remaining"
            )
            return None

        # === SMC ANALYSIS ===
        smc_state = self.smc.analyze(df)
        smc_bias = smc_state.bias_score
        smc_trend = smc_state.trend

        # Log state
        logger.info(
            f"[*] SQUEEZE: ON={sqz_on} | Fired={squeeze_fired} | Mom={mom:.2f} | Prev={mom_prev:.2f} "
            f"| EMA={ema_trend:.0f} | SMC={smc_trend}({smc_bias:+.0f}) "
            f"| Vol={vol:.0f}/{vol_avg:.0f}"
        )

        # === ENTRY LOGIC: SQUEEZE FIRE, MOMENTUM ACCEL, or TREND CONTINUATION ===
        signal_type = None
        direction = None

        # Pre-compute momentum history
        mom_prev2 = df.iloc[-3]["sqz_mom"] if len(df) > 2 else np.nan
        mom_prev3 = df.iloc[-4]["sqz_mom"] if len(df) > 3 else np.nan

        if squeeze_fired:
            # Type A: Squeeze fire
            if mom > 0:
                direction = "LONG"
            elif mom < 0:
                direction = "SHORT"
            else:
                logger.info("    Squeeze fired but momentum is zero")
                return None
            signal_type = "SQUEEZE_FIRE"

        elif sqz_on and not pd.isna(mom_prev):
            # Type B: Momentum acceleration while squeeze is building
            if not pd.isna(mom_prev2):
                mom_accel = mom - mom_prev
                mom_accel_prev = mom_prev - mom_prev2
                min_strong_mom = atr * 0.08  # Lowered from 0.15 — much more sensitive

                if (mom > min_strong_mom and mom_accel > 0 and mom_accel_prev > 0):
                    direction = "LONG"
                    signal_type = "MOM_ACCEL"
                elif (mom < -min_strong_mom and mom_accel < 0 and mom_accel_prev < 0):
                    direction = "SHORT"
                    signal_type = "MOM_ACCEL"

        if signal_type is None and not pd.isna(mom_prev2) and not pd.isna(mom_prev3):
            # Type C: Trend continuation — no squeeze needed
            # Strong momentum in same direction for 3+ bars aligned with EMA trend
            min_trend_mom = atr * 0.20
            all_positive = (mom > 0 and mom_prev > 0 and mom_prev2 > 0)
            all_negative = (mom < 0 and mom_prev < 0 and mom_prev2 < 0)

            if all_positive and mom > min_trend_mom and price > ema_trend:
                direction = "LONG"
                signal_type = "TREND_CONT"
            elif all_negative and abs(mom) > min_trend_mom and price < ema_trend:
                direction = "SHORT"
                signal_type = "TREND_CONT"

        if signal_type is None:
            logger.info("    No entry signal")
            return None

        # === FILTER 1: EMA TREND ALIGNMENT (with tolerance) ===
        ema_tolerance = ema_trend * self.ema_tolerance_pct
        if direction == "LONG" and price < (ema_trend - ema_tolerance):
            logger.info(f"    [BLOCKED] LONG rejected: price {price:.0f} < EMA {ema_trend:.0f} - tol")
            return None
        if direction == "SHORT" and price > (ema_trend + ema_tolerance):
            logger.info(f"    [BLOCKED] SHORT rejected: price {price:.0f} > EMA {ema_trend:.0f} + tol")
            return None

        # === FILTER 2: SMC BIAS CONFIRMATION ===
        # Don't trade against very strong SMC bias (loosened thresholds)
        smc_block_threshold = -60 if signal_type == "SQUEEZE_FIRE" else -50
        if direction == "LONG" and smc_bias < smc_block_threshold:
            logger.info(f"    [BLOCKED] LONG rejected: SMC bias strongly bearish ({smc_bias:+.0f})")
            return None
        if direction == "SHORT" and smc_bias > abs(smc_block_threshold):
            logger.info(f"    [BLOCKED] SHORT rejected: SMC bias strongly bullish ({smc_bias:+.0f})")
            return None

        # === FILTER 3: VOLUME CONFIRMATION (squeeze fire only) ===
        if signal_type == "SQUEEZE_FIRE" and vol_avg > 0 and vol < vol_avg * self.volume_confirm_mult:
            logger.info(
                f"    [BLOCKED] Volume too low: {vol:.0f} < {vol_avg * self.volume_confirm_mult:.0f}"
            )
            return None

        # === FILTER 4: MOMENTUM STRENGTH (squeeze fire only) ===
        if signal_type == "SQUEEZE_FIRE":
            min_mom = atr * 0.01  # Lowered from 0.02 — accept weaker squeezes
            if abs(mom) < min_mom:
                logger.info(f"    [BLOCKED] Momentum too weak: |{mom:.2f}| < {min_mom:.2f}")
                return None

        # === BUILD TRADE ===
        if signal_type == "SQUEEZE_FIRE":
            reason_parts = [f"SQUEEZE FIRE {direction} (mom={mom:.1f})"]
        elif signal_type == "MOM_ACCEL":
            reason_parts = [f"MOM ACCEL {direction} (mom={mom:.1f}, accel={mom - mom_prev:.1f})"]
        else:  # TREND_CONT
            reason_parts = [f"TREND CONT {direction} (mom={mom:.1f}, 3-bar streak)"]

        # Add SMC context to reason
        if smc_bias != 0:
            reason_parts.append(f"SMC={smc_trend}({smc_bias:+.0f})")

        # Check if price is near an active order block (bonus confluence)
        for ob in smc_state.active_order_blocks[-3:]:
            if direction == "LONG" and ob.direction == "bullish":
                if ob.bottom <= price <= ob.top * 1.005:
                    reason_parts.append("+ Bullish OB")
                    break
            elif direction == "SHORT" and ob.direction == "bearish":
                if ob.bottom * 0.995 <= price <= ob.top:
                    reason_parts.append("+ Bearish OB")
                    break

        reason = " | ".join(reason_parts)

        # Adjust R:R based on signal type
        if signal_type == "SQUEEZE_FIRE":
            sl_mult = self.sl_atr_mult
            tp_mult = self.tp_atr_mult
        elif signal_type == "MOM_ACCEL":
            sl_mult = self.sl_atr_mult * 0.85
            tp_mult = self.tp_atr_mult * 0.67  # ~1:2.0 R:R
        else:  # TREND_CONT — tightest SL, quick profit
            sl_mult = self.sl_atr_mult * 0.7
            tp_mult = self.tp_atr_mult * 0.50  # ~1:1.8 R:R

        if direction == "LONG":
            stop_loss = price - (atr * sl_mult)
            take_profit = price + (atr * tp_mult)
            signal = Signal.LONG
        else:
            stop_loss = price + (atr * sl_mult)
            take_profit = price - (atr * tp_mult)
            signal = Signal.SHORT

        position_size = self.calculate_position_size(
            account_balance, price, stop_loss
        )

        logger.info(f"[{'+'if direction=='LONG' else '-'}] {reason}")
        logger.info(
            f"    R:R = 1:{self.tp_atr_mult/self.sl_atr_mult:.1f} | "
            f"SL={stop_loss:.2f} | TP={take_profit:.2f} | Size={position_size:.6f}"
        )

        return TradeSignal(
            signal=signal,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            reason=reason,
        )


Strategy = SqzMomSmcStrategy
