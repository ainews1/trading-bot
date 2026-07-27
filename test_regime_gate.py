"""Focused test of TradingBot._regime_gate — no exchange connection needed."""
from types import SimpleNamespace
from bot import TradingBot
from config import config

def snap(trend, regime, strength=0.10):
    return SimpleNamespace(trend_direction=trend, market_regime=regime,
                           trend_strength=strength)

def sig(side):
    return SimpleNamespace(signal=SimpleNamespace(value=side))

def gate(snapshot, side):
    fake = SimpleNamespace(last_snapshot=snapshot)
    return TradingBot._regime_gate(fake, sig(side))

# Ensure filter is on for the test
config.REGIME_FILTER_ENABLED = True
config.BLOCK_COUNTER_TREND = True
config.BLOCK_RANGING_REGIME = True
config.COUNTER_TREND_MIN_STRENGTH = 0.0

cases = [
    # (snapshot, side, should_block, note)
    (snap("BULL", "TRENDING"), "LONG",  False, "with-trend long in bull"),
    (snap("BEAR", "TRENDING"), "SHORT", False, "with-trend short in bear"),
    (snap("BEAR", "TRENDING"), "LONG",  True,  "counter-trend long in bear"),
    (snap("BULL", "TRENDING"), "SHORT", True,  "counter-trend short in bull"),
    (snap("BULL", "RANGING"),  "LONG",  True,  "long in ranging"),
    (snap("BEAR", "RANGING"),  "SHORT", True,  "short in ranging"),
    (snap("NEUTRAL", "BREAKOUT"), "LONG",  False, "neutral breakout long ok"),
    (snap("NEUTRAL", "REVERSAL"), "SHORT", False, "neutral reversal short ok"),
]

passed = 0
for s, side, expect_block, note in cases:
    reason = gate(s, side)
    blocked = reason is not None
    ok = blocked == expect_block
    passed += ok
    print(f"[{'OK ' if ok else 'FAIL'}] {note:38} side={side:5} "
          f"-> {'BLOCK: '+reason if blocked else 'allow'}")

# Fail-open checks
config.REGIME_FILTER_ENABLED = False
assert gate(snap("BEAR", "RANGING"), "LONG") is None, "disabled flag must allow"
config.REGIME_FILTER_ENABLED = True
assert gate(None, "LONG") is None, "no snapshot must allow (fail-open)"
print("[OK ] fail-open: disabled flag and missing snapshot both allow")

# Strength threshold check
config.COUNTER_TREND_MIN_STRENGTH = 0.20
assert gate(snap("BEAR", "TRENDING", strength=0.10), "LONG") is None, "weak trend not blocked"
assert gate(snap("BEAR", "TRENDING", strength=0.30), "LONG") is not None, "strong trend blocked"
print("[OK ] strength threshold gates counter-trend block")

print(f"\n{passed}/{len(cases)} table cases passed")
assert passed == len(cases), "table cases failed"
print("ALL TESTS PASSED")
