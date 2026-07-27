"""
Hypothesis test: does the bot perform worse when BTC price is downtrending?
Parses trading_bot.log, reconstructs each closed paper trade, classifies the
BTC trend regime at exit two independent ways, and compares performance.
"""
import re, statistics
from datetime import datetime, timedelta

TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")

def parse_ts(line):
    m = TS.match(line)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")

price_series = []      # (ts, price) from every "Current price:" line
trades = []            # closed trades
cur_price = None
cur_trend = None       # bot scout label: 'BULL'/'BEAR'
cur_side = None        # last opened side
cur_strat = None       # active strategy
cur_strength = 0.0     # scout trend strength %
cur_regime = None      # scout market regime: TRENDING/RANGING/BREAKOUT/REVERSAL

with open("trading_bot.log", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

i = 0
n = len(lines)
while i < n:
    line = lines[i]
    ts = parse_ts(line)

    m = re.search(r"Current price:\s*([\d.]+)", line)
    if m and ts:
        cur_price = float(m.group(1))
        price_series.append((ts, cur_price))

    m = re.search(r"Trend:\s*(BULL|BEAR|NEUTRAL)\s*\(Strength:\s*([\d.]+)%\)", line)
    if m:
        cur_trend = m.group(1)
        cur_strength = float(m.group(2))

    m = re.search(r"Market Regime:\s*([A-Z]+)", line)
    if m:
        cur_regime = m.group(1)

    m = re.search(r"Opened (BUY|SELL|LONG|SHORT) position", line)
    if m:
        cur_side = {"BUY": "LONG", "SELL": "SHORT"}.get(m.group(1), m.group(1))

    m = re.search(r"Strategy:\s*([A-Z_]+)", line)
    if m:
        cur_strat = m.group(1)

    if "Position closed" in line and ts:
        reason = "TAKE PROFIT" if "TAKE PROFIT" in line else ("STOP LOSS" if "STOP LOSS" in line else "OTHER")
        pnl = None
        # PnL is on the next few lines
        for j in range(i+1, min(i+5, n)):
            pm = re.search(r"PnL:\s*\$(-?[\d.]+)", lines[j])
            if pm:
                pnl = float(pm.group(1))
                break
        if pnl is not None:
            trades.append({
                "ts": ts, "pnl": pnl, "reason": reason,
                "side": cur_side, "scout": cur_trend, "price": cur_price,
                "strat": cur_strat, "strength": cur_strength, "mregime": cur_regime,
            })
    i += 1

print(f"Raw closed-trade records: {len(trades)}")

# --- Dedupe: identical (ts-to-second, pnl, reason) collapsed (log replay artifact) ---
seen = set()
deduped = []
for t in trades:
    key = (t["ts"].replace(microsecond=0), round(t["pnl"], 2), t["reason"], t["side"])
    if key in seen:
        continue
    seen.add(key)
    deduped.append(t)
trades = deduped
print(f"After dedupe: {len(trades)}")
print(f"Trade date span: {trades[0]['ts'].date()} -> {trades[-1]['ts'].date()}")
print()

# --- Independent BTC trend: trailing 24h price slope at each trade's exit ---
price_series.sort(key=lambda x: x[0])
def price_at_or_before(target):
    # last price at or before target time
    lo, hi, ans = 0, len(price_series)-1, None
    while lo <= hi:
        mid = (lo+hi)//2
        if price_series[mid][0] <= target:
            ans = price_series[mid][1]; lo = mid+1
        else:
            hi = mid-1
    return ans

for t in trades:
    past = price_at_or_before(t["ts"] - timedelta(hours=24))
    now = t["price"] if t["price"] else price_at_or_before(t["ts"])
    if past and now:
        chg = (now - past) / past * 100
        t["btc_24h_chg"] = chg
        t["regime"] = "UP" if chg > 0.5 else ("DOWN" if chg < -0.5 else "FLAT")
    else:
        t["btc_24h_chg"] = None
        t["regime"] = "UNKNOWN"

def report(name, group):
    if not group:
        print(f"  {name}: no trades"); return
    pnls = [t["pnl"] for t in group]
    wins = [p for p in pnls if p > 0]
    total = sum(pnls)
    wr = len(wins)/len(pnls)*100
    exp = total/len(pnls)
    print(f"  {name:>10}: n={len(pnls):>3} | win%={wr:5.1f} | total=${total:8.2f} | "
          f"avg/trade=${exp:6.2f} | wins={len(wins)} losses={len(pnls)-len(wins)}")

from collections import Counter
print("Trades by strategy:", dict(Counter(t["strat"] for t in trades)))
print()

def regime_block(title, subset):
    print(f"### {title}  (n={len(subset)})")
    for r in ["UP", "FLAT", "DOWN"]:
        report(r, [t for t in subset if t["regime"] == r])
    for r in ["BULL", "BEAR"]:
        report("scout:"+r, [t for t in subset if t["scout"] == r])
    for s in ["LONG", "SHORT"]:
        report("side:"+s, [t for t in subset if t["side"] == s])
    print()

print("=== ALL TRADES (mixed strategies — confounded, context only) ===")
for r in ["UP", "FLAT", "DOWN", "UNKNOWN"]:
    report(r, [t for t in trades if t["regime"] == r])
print()

# The user's live bot is SQZMOM_SMC; the ~1 month they describe is recent activity.
sqz = [t for t in trades if t["strat"] == "SQZMOM_SMC"]
regime_block("SQZMOM_SMC ONLY (the live bot)", sqz)

recent = [t for t in trades if t["ts"] >= trades[-1]["ts"] - timedelta(days=35)]
regime_block("LAST 35 DAYS (all strategies)", recent)

sqz_recent = [t for t in sqz if t["ts"] >= trades[-1]["ts"] - timedelta(days=35)]
regime_block("SQZMOM_SMC + LAST 35 DAYS", sqz_recent)

print("=== (legacy view) Bucketed by BTC trailing-24h price trend, all trades ===")
for r in ["UP", "FLAT", "DOWN", "UNKNOWN"]:
    report(r, [t for t in trades if t["regime"] == r])

print()
print("=== Bucketed by bot's own scout Trend label at exit ===")
for r in ["BULL", "BEAR", None]:
    report(str(r), [t for t in trades if t["scout"] == r])

print()
print("=== By position side ===")
for s in ["LONG", "SHORT", None]:
    report(str(s), [t for t in trades if t["side"] == s])

print()
print("=== LONG trades split by BTC trend (the key cross-tab) ===")
for r in ["UP", "DOWN"]:
    report(f"LONG/{r}", [t for t in trades if t["side"]=="LONG" and t["regime"]==r])
for r in ["UP", "DOWN"]:
    report(f"SHORT/{r}", [t for t in trades if t["side"]=="SHORT" and t["regime"]==r])

# correlation between btc 24h change and pnl
print("\n=== COUNTERFACTUAL: effect of the proposed entry gate ===")
print("(uses scout trend/regime label at exit as proxy for entry; in-sample estimate)")
print("Regime distribution:", dict(Counter(t["mregime"] for t in trades)))

def counterfactual(label, block_fn, subset):
    kept = [t for t in subset if not block_fn(t)]
    blocked = [t for t in subset if block_fn(t)]
    base_total = sum(t["pnl"] for t in subset)
    kept_total = sum(t["pnl"] for t in kept)
    blk_total = sum(t["pnl"] for t in blocked)
    kept_exp = kept_total/len(kept) if kept else 0
    base_exp = base_total/len(subset) if subset else 0
    print(f"\n[{label}]")
    print(f"  baseline : n={len(subset)} total=${base_total:8.2f} exp=${base_exp:+.2f}/trade")
    print(f"  blocked  : n={len(blocked):>3} total=${blk_total:8.2f}  (P&L removed)")
    print(f"  kept     : n={len(kept):>3} total=${kept_total:8.2f} exp=${kept_exp:+.2f}/trade")
    print(f"  --> expectancy change: ${base_exp:+.2f} -> ${kept_exp:+.2f}/trade")

ct_only      = lambda t: (t["side"]=="LONG" and t["scout"]=="BEAR") or (t["side"]=="SHORT" and t["scout"]=="BULL")
ranging_only = lambda t: t["mregime"]=="RANGING"
both         = lambda t: ct_only(t) or ranging_only(t)

counterfactual("Block counter-trend only", ct_only, trades)
counterfactual("Block RANGING regime only", ranging_only, trades)
counterfactual("Block BOTH (proposed)", both, trades)
counterfactual("Block BOTH — last 35 days", both, recent)

# --- Temporal robustness: does the SAME rule help on a held-out later slice? ---
print("\n=== TEMPORAL SPLIT (walk-forward sanity: rule is fixed, not tuned) ===")
def winrate(g):
    return (sum(1 for t in g if t["pnl"] > 0) / len(g) * 100) if g else 0
def seg_report(name, subset):
    kept = [t for t in subset if not both(t)]
    print(f"  {name:14} | BASE n={len(subset):>3} win%={winrate(subset):4.1f} "
          f"exp=${sum(t['pnl'] for t in subset)/len(subset):+.2f}"
          f"   ->  FILTERED n={len(kept):>3} win%={winrate(kept):4.1f} "
          f"exp=${(sum(t['pnl'] for t in kept)/len(kept)) if kept else 0:+.2f}")
cut = int(len(trades) * 0.6)
seg_report("first 60%", trades[:cut])
seg_report("last 40%",  trades[cut:])
seg_report("ALL",       trades)

valid = [(t["btc_24h_chg"], t["pnl"]) for t in trades if t["btc_24h_chg"] is not None]
if len(valid) > 2:
    xs = [v[0] for v in valid]; ys = [v[1] for v in valid]
    try:
        r = statistics.correlation(xs, ys)
        print(f"\nPearson corr(BTC 24h %chg, trade PnL) = {r:+.3f}  (n={len(valid)})")
    except Exception as e:
        print("corr err", e)
