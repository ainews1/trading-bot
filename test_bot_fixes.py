"""
Tests for bot.py fixes (2026-06-10):
1. fetch_ohlcv drops the still-forming partial candle (closed candles only)
2. Daily loss limit measured against day-OPENING balance, not current balance
3. daily_open_balance persisted / reconstructed across restarts

Runs in a temp directory so the live paper_state.json and trading_bot.log
are never touched.

Run: python test_bot_fixes.py
"""

import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone

# Chdir BEFORE importing bot so its FileHandler + STATE_FILE land in temp dir
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = tempfile.mkdtemp(prefix="bot_fix_test_")
os.chdir(WORK_DIR)
sys.path.insert(0, PROJECT_DIR)

from bot import TradingBot, STATE_FILE  # noqa: E402

passed = 0


def check(label, cond):
    global passed
    assert cond, f"FAIL: {label}"
    passed += 1
    print(f"[OK ] {label}")


class StubExchange:
    def __init__(self, ohlcv):
        self._ohlcv = ohlcv

    def fetch_ohlcv(self, symbol, timeframe, limit=250):
        return self._ohlcv


INTERVAL = TradingBot._timeframe_seconds()  # follows config.TIMEFRAME


def make_candles(n_closed, include_forming):
    """Candles on the real config-timeframe grid ending at the current interval."""
    boundary = (int(time.time()) // INTERVAL) * INTERVAL
    rows = []
    for i in range(n_closed, 0, -1):
        ts = (boundary - INTERVAL * i) * 1000
        rows.append([ts, 100.0, 101.0, 99.0, 100.5, 10.0])
    if include_forming:
        rows.append([boundary * 1000, 100.5, 100.6, 100.4, 100.5, 0.3])
    return rows


# --- 1. Partial-candle handling -------------------------------------------
bot = TradingBot()

bot.exchange = StubExchange(make_candles(3, include_forming=True))
df = bot.fetch_ohlcv()
boundary_ts = (int(time.time()) // INTERVAL) * INTERVAL
check("forming candle dropped (4 rows -> 3)", df is not None and len(df) == 3)
check("last candle is closed (open < current boundary)",
      df.index[-1].timestamp() < boundary_ts)

bot.exchange = StubExchange(make_candles(3, include_forming=False))
df = bot.fetch_ohlcv()
check("all-closed fetch untouched (3 rows kept)", df is not None and len(df) == 3)

bot.exchange = StubExchange(make_candles(0, include_forming=True))
check("only-forming-candle fetch returns None", bot.fetch_ohlcv() is None)

bot.exchange = StubExchange([])
check("empty OHLCV response returns None", bot.fetch_ohlcv() is None)

# --- 2. Daily loss pct vs opening balance ----------------------------------
bot.daily_open_balance = 1000.0
bot.daily_pnl = -75.0
# old math: 75/925 = 8.1% would trip the 8% cap early; correct: 75/1000 = 7.5%
check("loss pct uses opening balance (7.5%, not 8.1%)",
      abs(bot._daily_loss_pct(balance=925.0) - 0.075) < 1e-12)

bot.daily_pnl = 40.0
check("profitable day -> 0 loss pct", bot._daily_loss_pct(925.0) == 0.0)

bot.daily_open_balance = 0.0
bot.daily_pnl = -75.0
check("missing opening balance falls back to current balance",
      abs(bot._daily_loss_pct(925.0) - 75.0 / 925.0) < 1e-12)

# --- 3. State persistence ---------------------------------------------------
bot.paper_balance = 925.0
bot.daily_pnl = -75.0
bot.daily_open_balance = 1000.0
bot.daily_pnl_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
bot._save_state()

bot2 = TradingBot()
check("daily_open_balance survives restart (same day)",
      bot2.daily_open_balance == 1000.0)
check("daily_pnl survives restart", bot2.daily_pnl == -75.0)

# Backward compat: state file from before this fix (no daily_open_balance key)
with open(STATE_FILE) as f:
    state = json.load(f)
del state['daily_open_balance']
with open(STATE_FILE, 'w') as f:
    json.dump(state, f)

bot3 = TradingBot()
check("legacy state reconstructs opening balance (balance - daily_pnl = 1000)",
      bot3.daily_open_balance == 1000.0)

print(f"\nALL {passed} TESTS PASSED")
