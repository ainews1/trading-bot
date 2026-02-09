# 🐕 Trading Bot - Bulldog Strategy

BTC/USDT perpetual futures scalping bot with dual strategy support.

## Strategies

### 1. Bulldog Pattern (New!)
Detects the "Bulldog" double-bottom reversal pattern:
- Back legs = First low
- Body/Back = Curved bounce
- Front legs = Second low (double bottom)
- Neck = Push up
- Head = Shallow pullback (25-38.2%)

Entry on head pullback or breakout. Uses Fibonacci extensions for targets.

### 2. EMA + RSI (Legacy)
Classic trend-following with momentum:
- Long: Price > EMA + RSI oversold
- Short: Price < EMA + RSI overbought

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
```

## Configuration

Edit `config.py`:
- `STRATEGY = "bulldog"` or `"ema_rsi"`
- `PAPER_TRADING = True` (keep True until ready!)
- `LEVERAGE = 5`

## Run

```bash
python bot.py
```

## Files

| File | Description |
|------|-------------|
| `bot.py` | Main trading loop |
| `strategy.py` | EMA+RSI strategy |
| `strategy_bulldog.py` | Bulldog pattern detector |
| `config.py` | All settings |
| `backtest*.py` | Backtesting tools |

## ⚠️ Disclaimer

Trading futures is risky. Only trade with money you can afford to lose.
