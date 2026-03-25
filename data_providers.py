"""
Advanced Data Providers
=======================
- Hyperliquid: Order book, OI, Funding
- TradingView: Webhook alerts
- Liquidation Heatmap: Coinglass/Binance
"""

import requests
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime
import time
import json

logger = logging.getLogger(__name__)


# ============================================================
# HYPERLIQUID API
# ============================================================
class HyperliquidAPI:
    """
    Hyperliquid DEX API - Free, no key needed
    Docs: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api
    """
    
    BASE_URL = "https://api.hyperliquid.xyz/info"
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 10  # seconds
    
    def _post(self, payload: dict) -> dict:
        try:
            resp = requests.post(self.BASE_URL, json=payload, timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning(f"Hyperliquid API error: {e}")
        return {}
    
    def get_meta(self) -> dict:
        """Get metadata for all assets"""
        return self._post({"type": "meta"})
    
    def get_all_mids(self) -> dict:
        """Get mid prices for all assets"""
        return self._post({"type": "allMids"})
    
    def get_l2_book(self, coin: str = "BTC") -> dict:
        """
        Get Level 2 order book
        Returns: {levels: [[price, size, numOrders], ...]}
        """
        cache_key = f"l2_{coin}"
        if cache_key in self.cache:
            data, ts = self.cache[cache_key]
            if time.time() - ts < self.cache_ttl:
                return data
        
        data = self._post({"type": "l2Book", "coin": coin})
        self.cache[cache_key] = (data, time.time())
        return data
    
    def get_funding_history(self, coin: str = "BTC", start_time: int = None) -> list:
        """Get funding rate history"""
        payload = {"type": "fundingHistory", "coin": coin}
        if start_time:
            payload["startTime"] = start_time
        return self._post(payload)
    
    def get_open_interest(self, coin: str = "BTC") -> Optional[float]:
        """Get open interest for a coin"""
        meta = self.get_meta()
        if meta and "universe" in meta:
            for asset in meta["universe"]:
                if asset.get("name") == coin:
                    # OI is in the asset info
                    return asset.get("openInterest")
        return None
    
    def get_orderbook_imbalance(self, coin: str = "BTC", depth: int = 10) -> dict:
        """
        Calculate order book imbalance
        Returns: {bid_volume, ask_volume, imbalance_ratio, bid_wall, ask_wall}
        """
        book = self.get_l2_book(coin)
        
        result = {
            "bid_volume": 0,
            "ask_volume": 0,
            "imbalance_ratio": 1.0,
            "bid_wall": None,
            "ask_wall": None,
            "spread": None
        }
        
        if not book or "levels" not in book:
            return result
        
        levels = book["levels"]
        if len(levels) < 2:
            return result
        
        bids = levels[0][:depth] if len(levels[0]) >= depth else levels[0]
        asks = levels[1][:depth] if len(levels[1]) >= depth else levels[1]
        
        # Calculate volumes
        bid_vol = sum(float(level["sz"]) for level in bids) if bids else 0
        ask_vol = sum(float(level["sz"]) for level in asks) if asks else 0
        
        result["bid_volume"] = bid_vol
        result["ask_volume"] = ask_vol
        
        if ask_vol > 0:
            result["imbalance_ratio"] = bid_vol / ask_vol
        
        # Find walls (large orders)
        if bids:
            max_bid = max(bids, key=lambda x: float(x["sz"]))
            result["bid_wall"] = {"price": float(max_bid["px"]), "size": float(max_bid["sz"])}
        
        if asks:
            max_ask = max(asks, key=lambda x: float(x["sz"]))
            result["ask_wall"] = {"price": float(max_ask["px"]), "size": float(max_ask["sz"])}
        
        # Spread
        if bids and asks:
            best_bid = float(bids[0]["px"])
            best_ask = float(asks[0]["px"])
            result["spread"] = (best_ask - best_bid) / best_bid * 100
        
        return result


# ============================================================
# LIQUIDATION DATA
# ============================================================
class LiquidationTracker:
    """
    Track liquidation levels and recent liquidations
    Uses Binance API for liquidation data
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 60
    
    def get_liquidation_levels(self, current_price: float) -> dict:
        """
        Estimate liquidation clusters based on common leverage levels
        Returns estimated liquidation zones
        """
        levels = {
            "long_liquidations": [],
            "short_liquidations": []
        }
        
        # Common leverage levels and their liquidation distances
        leverages = [5, 10, 20, 50, 100]
        
        for lev in leverages:
            # Long liquidation = entry - (entry / leverage) approximately
            # Simplified: at 10x, ~10% drop = liquidation
            liq_pct = 1 / lev
            
            long_liq = current_price * (1 - liq_pct * 0.8)  # 80% of theoretical
            short_liq = current_price * (1 + liq_pct * 0.8)
            
            levels["long_liquidations"].append({
                "leverage": lev,
                "price": long_liq,
                "distance_pct": (current_price - long_liq) / current_price * 100
            })
            
            levels["short_liquidations"].append({
                "leverage": lev,
                "price": short_liq,
                "distance_pct": (short_liq - current_price) / current_price * 100
            })
        
        return levels
    
    def get_liquidation_heatmap(self, symbol: str = "BTCUSDT") -> dict:
        """
        Get liquidation heatmap data from Binance
        Shows where liquidations are likely clustered
        """
        cache_key = f"liq_heat_{symbol}"
        if cache_key in self.cache:
            data, ts = self.cache[cache_key]
            if time.time() - ts < self.cache_ttl:
                return data
        
        try:
            # Get recent liquidations from Binance
            url = f"https://fapi.binance.com/fapi/v1/forceOrders?symbol={symbol}&limit=100"
            resp = requests.get(url, timeout=10)
            
            if resp.status_code == 200:
                liquidations = resp.json()
                
                # Aggregate by price zones
                long_liqs = {}
                short_liqs = {}
                
                for liq in liquidations:
                    price = float(liq["price"])
                    qty = float(liq["origQty"])
                    side = liq["side"]
                    
                    # Round to nearest $100 for BTC
                    price_zone = round(price / 100) * 100
                    
                    if side == "SELL":  # Long position liquidated
                        long_liqs[price_zone] = long_liqs.get(price_zone, 0) + qty
                    else:  # Short position liquidated
                        short_liqs[price_zone] = short_liqs.get(price_zone, 0) + qty
                
                result = {
                    "long_liquidations": sorted(long_liqs.items(), key=lambda x: x[1], reverse=True)[:10],
                    "short_liquidations": sorted(short_liqs.items(), key=lambda x: x[1], reverse=True)[:10],
                    "total_long_liq": sum(long_liqs.values()),
                    "total_short_liq": sum(short_liqs.values()),
                }
                
                self.cache[cache_key] = (result, time.time())
                return result
                
        except Exception as e:
            logger.warning(f"Liquidation data fetch failed: {e}")
        
        return {"long_liquidations": [], "short_liquidations": [], "total_long_liq": 0, "total_short_liq": 0}


# ============================================================
# TRADINGVIEW WEBHOOK RECEIVER
# ============================================================
from flask import Flask, request, jsonify
import threading

class TradingViewWebhook:
    """
    Receive webhook alerts from TradingView
    Run as background server
    """
    
    def __init__(self, port: int = 5000, secret_key: str = None):
        self.port = port
        self.secret_key = secret_key
        self.app = Flask(__name__)
        self.alerts: List[Dict] = []
        self.max_alerts = 100
        self.callbacks = []
        
        self._setup_routes()
    
    def _setup_routes(self):
        @self.app.route('/webhook', methods=['POST'])
        def webhook():
            try:
                data = request.json or {}
                
                # Verify secret if configured
                if self.secret_key:
                    if data.get('secret') != self.secret_key:
                        return jsonify({"error": "Invalid secret"}), 401
                
                alert = {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": data.get("symbol", "BTC"),
                    "action": data.get("action", ""),  # BUY, SELL, CLOSE
                    "price": data.get("price"),
                    "message": data.get("message", ""),
                    "indicator": data.get("indicator", ""),
                    "timeframe": data.get("timeframe", ""),
                    "raw": data
                }
                
                self.alerts.append(alert)
                if len(self.alerts) > self.max_alerts:
                    self.alerts.pop(0)
                
                logger.info(f"📺 TradingView Alert: {alert['action']} {alert['symbol']} @ {alert['price']}")
                
                # Call registered callbacks
                for callback in self.callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                
                return jsonify({"status": "ok", "received": alert})
                
            except Exception as e:
                logger.error(f"Webhook error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/alerts', methods=['GET'])
        def get_alerts():
            return jsonify(self.alerts[-20:])
        
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({"status": "ok", "alerts_count": len(self.alerts)})
    
    def register_callback(self, callback):
        """Register a function to be called when alert received"""
        self.callbacks.append(callback)
    
    def get_latest_alert(self, symbol: str = None) -> Optional[Dict]:
        """Get most recent alert, optionally filtered by symbol"""
        for alert in reversed(self.alerts):
            if symbol is None or alert["symbol"].upper() == symbol.upper():
                return alert
        return None
    
    def start(self, background: bool = True):
        """Start the webhook server"""
        if background:
            thread = threading.Thread(
                target=lambda: self.app.run(host='0.0.0.0', port=self.port, debug=False, use_reloader=False),
                daemon=True
            )
            thread.start()
            logger.info(f"📺 TradingView webhook server started on port {self.port}")
        else:
            self.app.run(host='0.0.0.0', port=self.port)


# ============================================================
# AGGREGATED DATA PROVIDER
# ============================================================
@dataclass
class AggregatedMarketData:
    """Combined data from all sources"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Price
    price: float = 0.0
    
    # Order book (Hyperliquid)
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    orderbook_imbalance: float = 1.0
    bid_wall: Optional[Dict] = None
    ask_wall: Optional[Dict] = None
    spread_pct: float = 0.0
    
    # Open Interest
    open_interest: float = 0.0
    oi_change_1h: float = 0.0
    
    # Funding
    funding_rate: float = 0.0
    
    # Liquidations
    long_liq_zones: List = field(default_factory=list)
    short_liq_zones: List = field(default_factory=list)
    recent_long_liqs: float = 0.0
    recent_short_liqs: float = 0.0
    
    # TradingView
    tv_signal: Optional[str] = None
    tv_alert: Optional[Dict] = None
    
    # Sentiment
    overall_bias: str = "NEUTRAL"
    confidence: float = 0.0


class MarketDataAggregator:
    """Aggregates data from all providers"""
    
    def __init__(self, tv_port: int = 5000, tv_secret: str = None):
        self.hyperliquid = HyperliquidAPI()
        self.liquidations = LiquidationTracker()
        self.tv_webhook = TradingViewWebhook(port=tv_port, secret_key=tv_secret)
        
        # Start TradingView webhook server
        self.tv_webhook.start(background=True)
    
    def get_all_data(self, symbol: str = "BTC", current_price: float = 0) -> AggregatedMarketData:
        """Fetch and aggregate all market data"""
        data = AggregatedMarketData()
        data.price = current_price
        
        # Hyperliquid order book
        try:
            ob = self.hyperliquid.get_orderbook_imbalance(symbol)
            data.bid_volume = ob["bid_volume"]
            data.ask_volume = ob["ask_volume"]
            data.orderbook_imbalance = ob["imbalance_ratio"]
            data.bid_wall = ob["bid_wall"]
            data.ask_wall = ob["ask_wall"]
            data.spread_pct = ob["spread"] or 0
        except Exception as e:
            logger.warning(f"Hyperliquid data error: {e}")
        
        # Liquidation data
        try:
            liq_levels = self.liquidations.get_liquidation_levels(current_price)
            data.long_liq_zones = liq_levels["long_liquidations"]
            data.short_liq_zones = liq_levels["short_liquidations"]
            
            liq_heat = self.liquidations.get_liquidation_heatmap(f"{symbol}USDT")
            data.recent_long_liqs = liq_heat["total_long_liq"]
            data.recent_short_liqs = liq_heat["total_short_liq"]
        except Exception as e:
            logger.warning(f"Liquidation data error: {e}")
        
        # TradingView alert
        tv_alert = self.tv_webhook.get_latest_alert(symbol)
        if tv_alert:
            data.tv_alert = tv_alert
            data.tv_signal = tv_alert.get("action")
        
        # Calculate overall bias
        signals = []
        
        # Order book imbalance signal
        if data.orderbook_imbalance > 1.3:
            signals.append(("BULLISH", 0.3))
        elif data.orderbook_imbalance < 0.7:
            signals.append(("BEARISH", 0.3))
        
        # Liquidation cascade signal
        if data.recent_long_liqs > data.recent_short_liqs * 2:
            signals.append(("BEARISH", 0.4))  # Long squeeze
        elif data.recent_short_liqs > data.recent_long_liqs * 2:
            signals.append(("BULLISH", 0.4))  # Short squeeze
        
        # TradingView signal
        if data.tv_signal:
            if data.tv_signal.upper() in ["BUY", "LONG"]:
                signals.append(("BULLISH", 0.5))
            elif data.tv_signal.upper() in ["SELL", "SHORT"]:
                signals.append(("BEARISH", 0.5))
        
        # Aggregate
        if signals:
            bull_score = sum(conf for bias, conf in signals if bias == "BULLISH")
            bear_score = sum(conf for bias, conf in signals if bias == "BEARISH")
            
            if bull_score > bear_score + 0.2:
                data.overall_bias = "BULLISH"
                data.confidence = bull_score
            elif bear_score > bull_score + 0.2:
                data.overall_bias = "BEARISH"
                data.confidence = bear_score
        
        return data


# Test
if __name__ == "__main__":
    print("Testing Data Providers...")
    
    # Test Hyperliquid
    print("\n=== HYPERLIQUID ===")
    hl = HyperliquidAPI()
    ob = hl.get_orderbook_imbalance("BTC")
    print(f"Bid Volume: {ob['bid_volume']:.2f}")
    print(f"Ask Volume: {ob['ask_volume']:.2f}")
    print(f"Imbalance: {ob['imbalance_ratio']:.2f}")
    if ob['bid_wall']:
        print(f"Bid Wall: {ob['bid_wall']['size']:.2f} @ ${ob['bid_wall']['price']:,.0f}")
    if ob['ask_wall']:
        print(f"Ask Wall: {ob['ask_wall']['size']:.2f} @ ${ob['ask_wall']['price']:,.0f}")
    
    # Test Liquidations
    print("\n=== LIQUIDATIONS ===")
    liq = LiquidationTracker()
    levels = liq.get_liquidation_levels(67000)
    print("Long Liquidation Zones:")
    for l in levels["long_liquidations"][:3]:
        print(f"  {l['leverage']}x: ${l['price']:,.0f} ({l['distance_pct']:.1f}% away)")
    
    heat = liq.get_liquidation_heatmap()
    print(f"\nRecent Long Liqs: {heat['total_long_liq']:.2f} BTC")
    print(f"Recent Short Liqs: {heat['total_short_liq']:.2f} BTC")
    
    print("\n=== TRADINGVIEW WEBHOOK ===")
    print("Webhook endpoint: POST http://localhost:5000/webhook")
    print("Payload format:")
    print('  {"symbol": "BTC", "action": "BUY", "price": 67000, "message": "EMA Cross"}')
