"""
Generate 4H Chart with Donchian Channel
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Load data
df = pd.read_csv('historical_data.csv', index_col='timestamp', parse_dates=True)

# Resample to 4H
df_4h = df.resample('4h').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
}).dropna()

# Last 60 candles (10 days)
df_4h = df_4h.tail(60)

# Donchian Channel
df_4h['high20'] = df_4h['high'].rolling(20).max()
df_4h['low20'] = df_4h['low'].rolling(20).min()
df_4h['ema50'] = df_4h['close'].ewm(span=50, adjust=False).mean()

# Plot
fig, ax = plt.subplots(figsize=(14, 7))

# Candlesticks
for idx, row in df_4h.iterrows():
    color = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
    ax.plot([idx, idx], [row['low'], row['high']], color=color, linewidth=1)
    ax.plot([idx, idx], [row['open'], row['close']], color=color, linewidth=4)

# Donchian Channel
ax.plot(df_4h.index, df_4h['high20'], 'b--', label='20-bar High', alpha=0.7)
ax.plot(df_4h.index, df_4h['low20'], 'r--', label='20-bar Low', alpha=0.7)
ax.fill_between(df_4h.index, df_4h['high20'], df_4h['low20'], alpha=0.1, color='blue')

# EMA
ax.plot(df_4h.index, df_4h['ema50'], 'orange', label='EMA 50', linewidth=2)

# Current price
current = df_4h['close'].iloc[-1]
ax.axhline(y=current, color='white', linestyle=':', alpha=0.5)
ax.text(df_4h.index[-1], current, f' ${current:,.0f}', fontsize=10, color='white', va='center')

ax.set_title('BTC/USDT 4H - Donchian Channel Strategy', fontsize=14, color='white')
ax.set_xlabel('Date', color='white')
ax.set_ylabel('Price (USDT)', color='white')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_facecolor('#1e1e1e')
fig.patch.set_facecolor('#1e1e1e')
ax.tick_params(colors='white')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

plt.tight_layout()
plt.savefig('chart_4h.png', dpi=150, facecolor='#1e1e1e')
print("Chart saved: chart_4h.png")
