# 🐕 Bulldog Trading Bot

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A cryptocurrency futures trading bot for BTC/USDT perpetual contracts using advanced pattern recognition and technical analysis strategies.

![Trading Bot](https://img.shields.io/badge/Trading-Bot-green?style=for-the-badge&logo=bitcoin)

## 🎯 Features

- **Bulldog Pattern Detection** - Novel double-bottom reversal pattern recognition
- **EMA + RSI Strategy** - Classic trend-following with momentum confirmation
- **Paper Trading Mode** - Safe testing without real funds
- **Risk Management** - Configurable stop-loss, take-profit, and position sizing
- **Backtesting Suite** - Historical performance analysis tools
- **Multi-Timeframe Support** - Works on 1m, 5m, 15m timeframes

## 📈 Strategies

### 🐕 Bulldog Pattern (Primary)

The Bulldog formation is a double-bottom reversal pattern inspired by DejaBrewTrading's technique:

```
        Neck ___/\____ Head
            /       \
    Back __/         \___ Entry Zone
        /
_______/
Back   First     Front
Legs   Low       Legs
       (Double Bottom)
```

**Pattern Components:**
- **Back Legs** - First swing low
- **Body/Back** - Curved bounce creating higher high
- **Front Legs** - Second low forming double bottom
- **Neck** - Push up from double bottom
- **Head** - Shallow pullback (25-38.2% max) = Entry zone

**Entry Signals:**
- Enter on head pullback (aggressive)
- Enter on breakout above neck (conservative)
- Uses Fibonacci extensions for profit targets

### 📊 EMA + RSI (Legacy)

Classic trend-following strategy:
- **Long**: Price above EMA + RSI oversold recovery
- **Short**: Price below EMA + RSI overbought rejection

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Poloniex Futures account (optional for paper trading)

### Installation

```bash
# Clone the repository
git clone https://github.com/ainews1/trading-bot.git
cd trading-bot

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Copy and edit configuration:
```python
# config.py
PAPER_TRADING = True  # Keep True until ready for live trading!
STRATEGY = "bulldog"  # or "ema_rsi"
LEVERAGE = 5
```

2. Set API keys (for live trading):
```bash
export POLONIEX_API_KEY="your_key"
export POLONIEX_API_SECRET="your_secret"
```

### Run

```bash
python bot.py
```

## 📁 Project Structure

```
trading-bot/
├── bot.py              # Main trading loop & execution
├── config.py           # Configuration settings
├── strategy.py         # EMA + RSI strategy
├── strategy_bulldog.py # Bulldog pattern detector
├── backtest.py         # Full backtesting engine
├── backtest_fast.py    # Optimized backtester
├── backtest_turbo.py   # Ultra-fast backtester
├── optimizer.py        # Parameter optimization
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## ⚙️ Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PAPER_TRADING` | `True` | Safe mode - no real trades |
| `STRATEGY` | `"bulldog"` | Strategy selection |
| `LEVERAGE` | `5` | Position leverage |
| `RISK_PER_TRADE` | `0.02` | 2% risk per trade |
| `STOP_LOSS_PCT` | `0.03` | 3% stop loss |
| `TAKE_PROFIT_PCT` | `0.02` | 2% take profit |
| `MAX_DAILY_LOSS` | `0.06` | 6% max daily drawdown |

### Bulldog-Specific Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BULLDOG_LOOKBACK` | `50` | Candles to scan |
| `BULLDOG_DOUBLE_BOTTOM_TOL` | `0.005` | 0.5% tolerance |
| `BULLDOG_MAX_PULLBACK` | `0.382` | 38.2% max retracement |
| `BULLDOG_MIN_PULLBACK` | `0.10` | 10% min retracement |

## 📊 Backtesting

Run historical analysis:

```bash
# Standard backtest
python backtest.py

# Fast backtest
python backtest_fast.py

# Parameter optimization
python optimizer.py
```

## 🛡️ Risk Management

The bot includes multiple safety features:

- **Paper Trading Mode** - Test strategies without risking real funds
- **Position Sizing** - Automatic calculation based on risk tolerance
- **Stop Losses** - Configurable stop-loss levels
- **Daily Loss Limits** - Automatic trading halt at max drawdown
- **Isolated Margin** - Limits risk to position size

## ⚠️ Disclaimer

**USE AT YOUR OWN RISK**

Trading cryptocurrency futures involves substantial risk of loss and is not suitable for all investors. The high degree of leverage can work against you as well as for you.

- Only trade with money you can afford to lose
- Past performance is not indicative of future results
- Always test thoroughly in paper trading mode first
- This software is provided "as-is" with no guarantees

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

*Built with ❤️ for algorithmic traders*
