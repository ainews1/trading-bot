"""
Telegram Trade Alerts
=====================
Sends trade open/close notifications to Telegram.
"""

import os
import logging
import urllib.request
import urllib.parse
import json

from config import config

logger = logging.getLogger(__name__)

# Distinguishes alerts when multiple bot instances share one Telegram chat
INSTANCE_TAG = getattr(config, "INSTANCE_TAG", "")


def send_alert(message: str) -> bool:
    """Send a message to Telegram. Returns True on success."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        logger.warning("Telegram not configured, skipping alert")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode(
        {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }
    ).encode("utf-8")

    try:
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            if not result.get("ok"):
                logger.error(f"Telegram API error: {result}")
                return False
        return True
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        return False


def trade_opened(
    side: str,
    entry: float,
    size: float,
    stop_loss: float,
    take_profit: float,
    balance: float,
    paper: bool = True,
):
    mode = f"[{INSTANCE_TAG}] PAPER" if paper else f"[{INSTANCE_TAG}] LIVE"
    emoji = "\U0001f7e2" if side == "buy" else "\U0001f534"
    msg = (
        f"{emoji} *{mode} TRADE OPENED*\n"
        f"Direction: *{side.upper()}*\n"
        f"Entry: `${entry:,.2f}`\n"
        f"Size: `{size:.6f} BTC`\n"
        f"Stop Loss: `${stop_loss:,.2f}`\n"
        f"Take Profit: `${take_profit:,.2f}`\n"
        f"Balance: `${balance:,.2f}`"
    )
    send_alert(msg)


def trade_closed(
    pnl: float,
    reason: str,
    balance: float,
    daily_pnl: float,
    total_pnl: float,
    paper: bool = True,
):
    mode = f"[{INSTANCE_TAG}] PAPER" if paper else f"[{INSTANCE_TAG}] LIVE"
    emoji = "\u2705" if pnl > 0 else "\u274c"
    msg = (
        f"{emoji} *{mode} TRADE CLOSED — {reason}*\n"
        f"PnL: `${pnl:+,.2f}`\n"
        f"Daily PnL: `${daily_pnl:+,.2f}`\n"
        f"Total PnL: `${total_pnl:+,.2f}`\n"
        f"Balance: `${balance:,.2f}`"
    )
    send_alert(msg)
