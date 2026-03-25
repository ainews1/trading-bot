"""
External Data Aggregator
========================
Fetches order flow data from multiple sources:
- Coinglass: OI, Funding, Liquidations
- Coinalyze: OI Delta
- Exchange: Order book depth
"""

import requests
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)


@dataclass
class OrderFlowData:
    """Aggregated order flow signals"""
    timestamp: datetime
    
    # Open Interest
    open_interest: Optional[float] = None
    oi_change_1h: Optional[float] = None  # % change
    oi_change_24h: Optional[float] = None
    
    # Funding Rate
    funding_rate: Optional[float] = None
    predicted_funding: Optional[float] = None
    
    # Liquidations (last 1h)
    long_liquidations: Optional[float] = None
    short_liquidations: Optional[float] = None
    liquidation_ratio: Optional[float] = None  # long/short
    
    # Volume Delta
    cvd_1h: Optional[float] = None  # Cumulative Volume Delta
    buy_volume: Optional[float] = None
    sell_volume: Optional[float] = None
    
    # Long/Short Ratio
    long_short_ratio: Optional[float] = None
    
    # Sentiment score (-100 to +100)
    sentiment_score: Optional[float] = None
    
    def calculate_sentiment(self):
        """Calculate overall sentiment from available data"""
        signals = []
        
        # Funding rate signal
        if self.funding_rate is not None:
            if self.funding_rate > 0.01:  # Very positive = bearish (crowded long)
                signals.append(-30)
            elif self.funding_rate > 0.005:
                signals.append(-15)
            elif self.funding_rate < -0.01:  # Very negative = bullish
                signals.append(30)
            elif self.funding_rate < -0.005:
                signals.append(15)
            else:
                signals.append(0)
        
        # OI change signal
        if self.oi_change_1h is not None:
            if self.oi_change_1h > 5:  # Big OI increase
                signals.append(20)  # Could be bullish buildup
            elif self.oi_change_1h < -5:  # Big OI decrease
                signals.append(-10)  # Liquidations/closing
        
        # Liquidation signal
        if self.liquidation_ratio is not None:
            if self.liquidation_ratio > 2:  # More longs liquidated
                signals.append(-25)
            elif self.liquidation_ratio < 0.5:  # More shorts liquidated
                signals.append(25)
        
        # Long/Short ratio signal
        if self.long_short_ratio is not None:
            if self.long_short_ratio > 1.5:  # Crowded long
                signals.append(-20)
            elif self.long_short_ratio < 0.7:  # Crowded short
                signals.append(20)
        
        if signals:
            self.sentiment_score = sum(signals) / len(signals) * (len(signals) / 4)
        
        return self.sentiment_score


class ExternalDataFetcher:
    """Fetches data from external APIs"""
    
    def __init__(self, coinglass_api_key: str = None):
        self.coinglass_key = coinglass_api_key
        self.cache = {}
        self.cache_ttl = 60  # seconds
    
    def _get_cached(self, key: str) -> Optional[Any]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return data
        return None
    
    def _set_cache(self, key: str, data: Any):
        self.cache[key] = (data, time.time())
    
    def fetch_coinglass_oi(self, symbol: str = "BTC") -> Dict:
        """Fetch Open Interest from Coinglass"""
        cached = self._get_cached(f"cg_oi_{symbol}")
        if cached:
            return cached
        
        try:
            # Public endpoint (no key needed for basic data)
            url = f"https://open-api.coinglass.com/public/v2/open_interest?symbol={symbol}"
            headers = {}
            if self.coinglass_key:
                headers["coinglassSecret"] = self.coinglass_key
            
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                self._set_cache(f"cg_oi_{symbol}", data)
                return data
        except Exception as e:
            logger.warning(f"Coinglass OI fetch failed: {e}")
        
        return {}
    
    def fetch_coinglass_funding(self, symbol: str = "BTC") -> Dict:
        """Fetch Funding Rate from Coinglass"""
        cached = self._get_cached(f"cg_funding_{symbol}")
        if cached:
            return cached
        
        try:
            url = f"https://open-api.coinglass.com/public/v2/funding?symbol={symbol}"
            headers = {}
            if self.coinglass_key:
                headers["coinglassSecret"] = self.coinglass_key
            
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                self._set_cache(f"cg_funding_{symbol}", data)
                return data
        except Exception as e:
            logger.warning(f"Coinglass funding fetch failed: {e}")
        
        return {}
    
    def fetch_coinglass_liquidations(self, symbol: str = "BTC") -> Dict:
        """Fetch recent liquidations"""
        cached = self._get_cached(f"cg_liq_{symbol}")
        if cached:
            return cached
        
        try:
            url = f"https://open-api.coinglass.com/public/v2/liquidation_history?symbol={symbol}&time_type=h1"
            headers = {}
            if self.coinglass_key:
                headers["coinglassSecret"] = self.coinglass_key
            
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                self._set_cache(f"cg_liq_{symbol}", data)
                return data
        except Exception as e:
            logger.warning(f"Coinglass liquidations fetch failed: {e}")
        
        return {}
    
    def fetch_coinglass_long_short(self, symbol: str = "BTC") -> Dict:
        """Fetch Long/Short ratio"""
        cached = self._get_cached(f"cg_ls_{symbol}")
        if cached:
            return cached
        
        try:
            url = f"https://open-api.coinglass.com/public/v2/long_short?symbol={symbol}"
            headers = {}
            if self.coinglass_key:
                headers["coinglassSecret"] = self.coinglass_key
            
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                self._set_cache(f"cg_ls_{symbol}", data)
                return data
        except Exception as e:
            logger.warning(f"Coinglass L/S fetch failed: {e}")
        
        return {}
    
    def get_aggregated_data(self, symbol: str = "BTC") -> OrderFlowData:
        """Fetch all data and aggregate into OrderFlowData"""
        data = OrderFlowData(timestamp=datetime.now())
        
        # Fetch OI
        oi_data = self.fetch_coinglass_oi(symbol)
        if oi_data.get("data"):
            try:
                data.open_interest = float(oi_data["data"][0].get("openInterest", 0))
                data.oi_change_24h = float(oi_data["data"][0].get("h24Change", 0))
            except (KeyError, IndexError, TypeError):
                pass
        
        # Fetch Funding
        funding_data = self.fetch_coinglass_funding(symbol)
        if funding_data.get("data"):
            try:
                # Average across exchanges
                rates = [float(ex.get("rate", 0)) for ex in funding_data["data"] if ex.get("rate")]
                if rates:
                    data.funding_rate = sum(rates) / len(rates)
            except (KeyError, TypeError):
                pass
        
        # Fetch Liquidations
        liq_data = self.fetch_coinglass_liquidations(symbol)
        if liq_data.get("data"):
            try:
                data.long_liquidations = float(liq_data["data"].get("longLiquidationUsd", 0))
                data.short_liquidations = float(liq_data["data"].get("shortLiquidationUsd", 0))
                if data.short_liquidations > 0:
                    data.liquidation_ratio = data.long_liquidations / data.short_liquidations
            except (KeyError, TypeError):
                pass
        
        # Fetch Long/Short Ratio
        ls_data = self.fetch_coinglass_long_short(symbol)
        if ls_data.get("data"):
            try:
                data.long_short_ratio = float(ls_data["data"][0].get("longShortRatio", 1.0))
            except (KeyError, IndexError, TypeError):
                pass
        
        # Calculate sentiment
        data.calculate_sentiment()
        
        return data


def get_order_flow_signal(data: OrderFlowData) -> str:
    """
    Convert order flow data to trading signal filter
    Returns: 'BULLISH', 'BEARISH', or 'NEUTRAL'
    """
    if data.sentiment_score is None:
        return 'NEUTRAL'
    
    if data.sentiment_score > 15:
        return 'BULLISH'
    elif data.sentiment_score < -15:
        return 'BEARISH'
    else:
        return 'NEUTRAL'


class BinanceFetcher:
    """Fetch data directly from Binance Futures API (no key needed for public data)"""
    
    def __init__(self):
        self.base_url = "https://fapi.binance.com"
        self.cache = {}
        self.cache_ttl = 30
    
    def _get_cached(self, key: str) -> Optional[Any]:
        if key in self.cache:
            data, ts = self.cache[key]
            if time.time() - ts < self.cache_ttl:
                return data
        return None
    
    def _set_cache(self, key: str, data: Any):
        self.cache[key] = (data, time.time())
    
    def get_funding_rate(self, symbol: str = "BTCUSDT") -> Optional[float]:
        """Get current funding rate"""
        cached = self._get_cached(f"funding_{symbol}")
        if cached is not None:
            return cached
        
        try:
            url = f"{self.base_url}/fapi/v1/fundingRate?symbol={symbol}&limit=1"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    rate = float(data[0]["fundingRate"])
                    self._set_cache(f"funding_{symbol}", rate)
                    return rate
        except Exception as e:
            logger.warning(f"Binance funding fetch failed: {e}")
        return None
    
    def get_open_interest(self, symbol: str = "BTCUSDT") -> Optional[float]:
        """Get open interest"""
        cached = self._get_cached(f"oi_{symbol}")
        if cached is not None:
            return cached
        
        try:
            url = f"{self.base_url}/fapi/v1/openInterest?symbol={symbol}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                oi = float(data["openInterest"])
                self._set_cache(f"oi_{symbol}", oi)
                return oi
        except Exception as e:
            logger.warning(f"Binance OI fetch failed: {e}")
        return None
    
    def get_long_short_ratio(self, symbol: str = "BTCUSDT") -> Optional[float]:
        """Get top trader long/short ratio"""
        cached = self._get_cached(f"ls_{symbol}")
        if cached is not None:
            return cached
        
        try:
            url = f"{self.base_url}/futures/data/topLongShortAccountRatio?symbol={symbol}&period=1h&limit=1"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    ratio = float(data[0]["longShortRatio"])
                    self._set_cache(f"ls_{symbol}", ratio)
                    return ratio
        except Exception as e:
            logger.warning(f"Binance L/S fetch failed: {e}")
        return None
    
    def get_taker_buy_sell_volume(self, symbol: str = "BTCUSDT") -> tuple:
        """Get taker buy/sell volume ratio"""
        cached = self._get_cached(f"taker_{symbol}")
        if cached is not None:
            return cached
        
        try:
            url = f"{self.base_url}/futures/data/takerlongshortRatio?symbol={symbol}&period=1h&limit=1"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    buy_ratio = float(data[0]["buyVol"])
                    sell_ratio = float(data[0]["sellVol"])
                    self._set_cache(f"taker_{symbol}", (buy_ratio, sell_ratio))
                    return buy_ratio, sell_ratio
        except Exception as e:
            logger.warning(f"Binance taker volume fetch failed: {e}")
        return None, None
    
    def get_aggregated_data(self) -> OrderFlowData:
        """Fetch all data from Binance"""
        data = OrderFlowData(timestamp=datetime.now())
        
        data.funding_rate = self.get_funding_rate()
        data.open_interest = self.get_open_interest()
        data.long_short_ratio = self.get_long_short_ratio()
        
        buy_vol, sell_vol = self.get_taker_buy_sell_volume()
        data.buy_volume = buy_vol
        data.sell_volume = sell_vol
        
        data.calculate_sentiment()
        
        return data


# Quick test
if __name__ == "__main__":
    print("Testing Binance Futures API...")
    fetcher = BinanceFetcher()
    data = fetcher.get_aggregated_data()
    
    print("=" * 50)
    print("ORDER FLOW DATA (Binance)")
    print("=" * 50)
    print(f"Open Interest: {data.open_interest:,.2f} BTC" if data.open_interest else "OI: N/A")
    print(f"Funding Rate: {data.funding_rate*100:.4f}%" if data.funding_rate else "Funding: N/A")
    print(f"Long/Short Ratio: {data.long_short_ratio:.2f}" if data.long_short_ratio else "L/S Ratio: N/A")
    print(f"Buy Volume: {data.buy_volume:.2f}" if data.buy_volume else "Buy Vol: N/A")
    print(f"Sell Volume: {data.sell_volume:.2f}" if data.sell_volume else "Sell Vol: N/A")
    print(f"\nSentiment Score: {data.sentiment_score:+.1f}" if data.sentiment_score else "Sentiment: N/A")
    print(f"Signal: {get_order_flow_signal(data)}")
