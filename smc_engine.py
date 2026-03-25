"""
Smart Money Concepts (SMC) Engine
=================================
Core ICT concepts: BOS, CHoCH, Order Blocks, Fair Value Gaps, Liquidity Sweeps.
Reusable module — consumed by strategy_sqzmom_smc.py.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class SwingPoint:
    index: int
    price: float
    swing_type: str  # "high" or "low"
    label: str = ""  # "HH", "HL", "LH", "LL" (set after classification)


@dataclass
class StructureBreak:
    index: int
    price: float
    break_type: str   # "BOS" or "CHoCH"
    direction: str    # "bullish" or "bearish"
    broken_level: float


@dataclass
class OrderBlock:
    index: int
    top: float
    bottom: float
    direction: str  # "bullish" or "bearish"
    mitigated: bool = False
    age: int = 0


@dataclass
class FairValueGap:
    index: int
    top: float
    bottom: float
    direction: str  # "bullish" or "bearish"
    filled: bool = False


@dataclass
class LiquiditySweep:
    index: int
    swept_level: float
    direction: str  # "bullish" (swept lows, reversed up) or "bearish"


@dataclass
class SMCState:
    trend: str = "neutral"  # "bullish", "bearish", "neutral"
    swing_points: List[SwingPoint] = field(default_factory=list)
    structure_breaks: List[StructureBreak] = field(default_factory=list)
    active_order_blocks: List[OrderBlock] = field(default_factory=list)
    active_fvgs: List[FairValueGap] = field(default_factory=list)
    recent_sweeps: List[LiquiditySweep] = field(default_factory=list)
    bias_score: float = 0.0  # -100 (bearish) to +100 (bullish)


class SmartMoneyEngine:
    """Detects SMC patterns on OHLCV DataFrames."""

    def __init__(
        self,
        swing_lookback: int = 5,
        ob_max_age: int = 50,
        fvg_min_gap_pct: float = 0.001,
        sweep_threshold_pct: float = 0.002,
    ):
        self.swing_lookback = swing_lookback
        self.ob_max_age = ob_max_age
        self.fvg_min_gap_pct = fvg_min_gap_pct
        self.sweep_threshold_pct = sweep_threshold_pct

    def analyze(self, df: pd.DataFrame) -> SMCState:
        """Full SMC analysis. Returns SMCState with all detected patterns."""
        if len(df) < self.swing_lookback * 4:
            return SMCState()

        highs = df["high"].values
        lows = df["low"].values
        opens = df["open"].values
        closes = df["close"].values

        swings = self._find_swing_points(highs, lows)
        swings = self._classify_swings(swings)
        trend = self._determine_trend(swings)
        breaks = self._detect_structure_breaks(closes, swings, trend)
        obs = self._detect_order_blocks(opens, highs, lows, closes, breaks)
        obs = self._mitigate_order_blocks(obs, highs, lows, len(df))
        fvgs = self._detect_fvgs(highs, lows)
        fvgs = self._fill_fvgs(fvgs, highs, lows, len(df))
        sweeps = self._detect_liquidity_sweeps(highs, lows, closes, swings)
        bias = self._calculate_bias(breaks, obs, fvgs, sweeps, len(df))

        # Update trend from latest breaks
        if breaks:
            last_break = breaks[-1]
            if last_break.break_type == "CHoCH":
                trend = last_break.direction
            elif last_break.break_type == "BOS":
                trend = last_break.direction

        return SMCState(
            trend=trend,
            swing_points=swings,
            structure_breaks=breaks,
            active_order_blocks=[ob for ob in obs if not ob.mitigated],
            active_fvgs=[fvg for fvg in fvgs if not fvg.filled],
            recent_sweeps=sweeps[-5:] if sweeps else [],
            bias_score=bias,
        )

    # ------------------------------------------------------------------ swings
    def _find_swing_points(
        self, highs: np.ndarray, lows: np.ndarray
    ) -> List[SwingPoint]:
        swings: List[SwingPoint] = []
        lb = self.swing_lookback
        n = len(highs)

        for i in range(lb, n - lb):
            # Swing high: highest high in window
            window_highs = highs[i - lb : i + lb + 1]
            if highs[i] == window_highs.max() and np.sum(window_highs == highs[i]) == 1:
                swings.append(SwingPoint(index=i, price=float(highs[i]), swing_type="high"))

            # Swing low: lowest low in window
            window_lows = lows[i - lb : i + lb + 1]
            if lows[i] == window_lows.min() and np.sum(window_lows == lows[i]) == 1:
                swings.append(SwingPoint(index=i, price=float(lows[i]), swing_type="low"))

        swings.sort(key=lambda s: s.index)
        return swings

    def _classify_swings(self, swings: List[SwingPoint]) -> List[SwingPoint]:
        """Label swings as HH/HL/LH/LL based on prior swing of same type."""
        last_high: Optional[SwingPoint] = None
        last_low: Optional[SwingPoint] = None

        for s in swings:
            if s.swing_type == "high":
                if last_high is not None:
                    s.label = "HH" if s.price > last_high.price else "LH"
                else:
                    s.label = "HH"
                last_high = s
            else:
                if last_low is not None:
                    s.label = "HL" if s.price > last_low.price else "LL"
                else:
                    s.label = "HL"
                last_low = s
        return swings

    def _determine_trend(self, swings: List[SwingPoint]) -> str:
        recent = swings[-6:] if len(swings) >= 6 else swings
        hh = sum(1 for s in recent if s.label == "HH")
        hl = sum(1 for s in recent if s.label == "HL")
        lh = sum(1 for s in recent if s.label == "LH")
        ll = sum(1 for s in recent if s.label == "LL")

        bull = hh + hl
        bear = lh + ll
        if bull > bear:
            return "bullish"
        elif bear > bull:
            return "bearish"
        return "neutral"

    # --------------------------------------------------------- structure breaks
    def _detect_structure_breaks(
        self,
        closes: np.ndarray,
        swings: List[SwingPoint],
        initial_trend: str,
    ) -> List[StructureBreak]:
        breaks: List[StructureBreak] = []
        trend = initial_trend

        # Track latest swing high and low
        last_swing_high: Optional[SwingPoint] = None
        last_swing_low: Optional[SwingPoint] = None

        for s in swings:
            if s.swing_type == "high":
                last_swing_high = s
            else:
                last_swing_low = s

        if not last_swing_high or not last_swing_low:
            return breaks

        # Scan candles after the last few swings for breaks
        swing_highs = [s for s in swings if s.swing_type == "high"]
        swing_lows = [s for s in swings if s.swing_type == "low"]

        prev_high: Optional[SwingPoint] = None
        prev_low: Optional[SwingPoint] = None
        current_trend = initial_trend

        for i, s in enumerate(swings):
            if s.swing_type == "high":
                if prev_high is not None:
                    # Check candles between prev swing and this one for a close above prev high
                    start = prev_high.index + 1
                    end = min(s.index + 1, len(closes))
                    for j in range(start, end):
                        if closes[j] > prev_high.price:
                            if current_trend == "bullish" or current_trend == "neutral":
                                breaks.append(StructureBreak(
                                    index=j, price=float(closes[j]),
                                    break_type="BOS", direction="bullish",
                                    broken_level=prev_high.price,
                                ))
                            else:
                                breaks.append(StructureBreak(
                                    index=j, price=float(closes[j]),
                                    break_type="CHoCH", direction="bullish",
                                    broken_level=prev_high.price,
                                ))
                                current_trend = "bullish"
                            break
                prev_high = s
            else:
                if prev_low is not None:
                    start = prev_low.index + 1
                    end = min(s.index + 1, len(closes))
                    for j in range(start, end):
                        if closes[j] < prev_low.price:
                            if current_trend == "bearish" or current_trend == "neutral":
                                breaks.append(StructureBreak(
                                    index=j, price=float(closes[j]),
                                    break_type="BOS", direction="bearish",
                                    broken_level=prev_low.price,
                                ))
                            else:
                                breaks.append(StructureBreak(
                                    index=j, price=float(closes[j]),
                                    break_type="CHoCH", direction="bearish",
                                    broken_level=prev_low.price,
                                ))
                                current_trend = "bearish"
                            break
                prev_low = s

        return breaks

    # ------------------------------------------------------------- order blocks
    def _detect_order_blocks(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        breaks: List[StructureBreak],
    ) -> List[OrderBlock]:
        obs: List[OrderBlock] = []

        for brk in breaks:
            idx = brk.index
            if brk.direction == "bullish":
                # Walk back to find last bearish (red) candle before the impulse
                for j in range(idx - 1, max(idx - 15, -1), -1):
                    if j < 0:
                        break
                    if closes[j] < opens[j]:  # Red candle
                        obs.append(OrderBlock(
                            index=j,
                            top=float(highs[j]),
                            bottom=float(lows[j]),
                            direction="bullish",
                        ))
                        break
            else:
                # Walk back to find last bullish (green) candle
                for j in range(idx - 1, max(idx - 15, -1), -1):
                    if j < 0:
                        break
                    if closes[j] > opens[j]:  # Green candle
                        obs.append(OrderBlock(
                            index=j,
                            top=float(highs[j]),
                            bottom=float(lows[j]),
                            direction="bearish",
                        ))
                        break
        return obs

    def _mitigate_order_blocks(
        self, obs: List[OrderBlock], highs: np.ndarray, lows: np.ndarray, n: int
    ) -> List[OrderBlock]:
        for ob in obs:
            ob.age = n - 1 - ob.index
            if ob.age > self.ob_max_age:
                ob.mitigated = True
                continue
            # Check if price returned through the OB
            for j in range(ob.index + 1, n):
                if ob.direction == "bullish":
                    if lows[j] <= ob.bottom:
                        ob.mitigated = True
                        break
                else:
                    if highs[j] >= ob.top:
                        ob.mitigated = True
                        break
        return obs

    # ------------------------------------------------------- fair value gaps
    def _detect_fvgs(
        self, highs: np.ndarray, lows: np.ndarray
    ) -> List[FairValueGap]:
        fvgs: List[FairValueGap] = []
        n = len(highs)

        for i in range(2, n):
            mid_price = (highs[i] + lows[i]) / 2
            min_gap = mid_price * self.fvg_min_gap_pct

            # Bullish FVG: candle[i-2] high < candle[i] low (gap up)
            if lows[i] > highs[i - 2] and (lows[i] - highs[i - 2]) > min_gap:
                fvgs.append(FairValueGap(
                    index=i - 1,
                    top=float(lows[i]),
                    bottom=float(highs[i - 2]),
                    direction="bullish",
                ))

            # Bearish FVG: candle[i-2] low > candle[i] high (gap down)
            if highs[i] < lows[i - 2] and (lows[i - 2] - highs[i]) > min_gap:
                fvgs.append(FairValueGap(
                    index=i - 1,
                    top=float(lows[i - 2]),
                    bottom=float(highs[i]),
                    direction="bearish",
                ))
        return fvgs

    def _fill_fvgs(
        self, fvgs: List[FairValueGap], highs: np.ndarray, lows: np.ndarray, n: int
    ) -> List[FairValueGap]:
        for fvg in fvgs:
            for j in range(fvg.index + 2, n):
                if fvg.direction == "bullish":
                    if lows[j] <= fvg.bottom:
                        fvg.filled = True
                        break
                else:
                    if highs[j] >= fvg.top:
                        fvg.filled = True
                        break
        return fvgs

    # -------------------------------------------------------- liquidity sweeps
    def _detect_liquidity_sweeps(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        swings: List[SwingPoint],
    ) -> List[LiquiditySweep]:
        sweeps: List[LiquiditySweep] = []
        n = len(closes)

        for s in swings:
            threshold = s.price * self.sweep_threshold_pct

            if s.swing_type == "low":
                # Bullish sweep: wick below swing low, close above
                for j in range(s.index + 1, min(s.index + 20, n)):
                    if lows[j] < s.price - threshold and closes[j] > s.price:
                        sweeps.append(LiquiditySweep(
                            index=j, swept_level=s.price, direction="bullish",
                        ))
                        break

            elif s.swing_type == "high":
                # Bearish sweep: wick above swing high, close below
                for j in range(s.index + 1, min(s.index + 20, n)):
                    if highs[j] > s.price + threshold and closes[j] < s.price:
                        sweeps.append(LiquiditySweep(
                            index=j, swept_level=s.price, direction="bearish",
                        ))
                        break
        return sweeps

    # ------------------------------------------------------------------- bias
    def _calculate_bias(
        self,
        breaks: List[StructureBreak],
        obs: List[OrderBlock],
        fvgs: List[FairValueGap],
        sweeps: List[LiquiditySweep],
        n: int,
    ) -> float:
        score = 0.0
        recency_window = 20

        # Structure breaks (most important)
        for brk in breaks:
            if brk.index >= n - recency_window:
                weight = 30 if brk.break_type == "CHoCH" else 20
                score += weight if brk.direction == "bullish" else -weight

        # Active order blocks
        for ob in obs:
            if not ob.mitigated and ob.age < recency_window:
                score += 15 if ob.direction == "bullish" else -15

        # Active FVGs
        for fvg in fvgs:
            if not fvg.filled and fvg.index >= n - recency_window:
                score += 10 if fvg.direction == "bullish" else -10

        # Liquidity sweeps
        for sweep in sweeps:
            if sweep.index >= n - recency_window:
                score += 20 if sweep.direction == "bullish" else -20

        return max(-100, min(100, score))
