from typing import Dict, List, Any, Optional
import numpy as np
from .integrate_scalping import compute_spread_and_mid, orderbook_imbalance, nearest_wall_proximity

class AdvancedScalpingStrategy:
    """
    Microstructure-aware scalper with spread/imbalance/MA-gap + wall proximity and micro-slippage guards.
    """

    def __init__(self, config: Dict[str, Any], exchange):
        self.config = config or {}
        self.exchange = exchange

        self.max_spread = float(self.config.get("max_spread", 0.0006))   # 6 bps
        self.min_imbalance = float(self.config.get("min_imbalance", 0.20))
        self.ma_fast = int(self.config.get("ma_fast", 20))
        self.ma_slow = int(self.config.get("ma_slow", 200))
        self.ma_gap_bps = float(self.config.get("ma_gap_bps", 8.0))      # 8 bps between price and MA
        self.slippage_bps = float(self.config.get("slippage_bps", 7.0))  # 7 bps mid-move over last 3 ticks

    async def scan(self, market_data: Dict[str, Any], predictions: Dict[str, Any]) -> List[Dict]:
        signals: List[Dict] = []
        for symbol, md in market_data.items():
            sig = await self._analyze_symbol(symbol, md)
            if sig:
                signals.append(sig)
        return signals

    async def _analyze_symbol(self, symbol: str, md: Dict[str, Any]) -> Optional[Dict]:
        ohlcv = md.get("ohlcv") or []
        orderbook = md.get("orderbook") or {}
        ticker = md.get("ticker") or {}
        if len(ohlcv) < max(self.ma_slow + 2, 50) or not orderbook:
            return None

        spread, best_bid, best_ask = compute_spread_and_mid(orderbook)
        if spread <= 0 or spread > self.max_spread:
            return None

        # Current price
        px = float(ohlcv[-1][4])

        # Moving averages (simple)
        closes = np.array([c[4] for c in ohlcv])
        ma_f = closes[-self.ma_fast:].mean()
        ma_s = closes[-self.ma_slow:].mean()

        # Require a minimal "gap" to MA to avoid trading noise
        gap_bps = abs(px - ma_f) / px * 1e4
        if gap_bps < self.ma_gap_bps:
            return None

        imb = orderbook_imbalance(orderbook, depth=5)

        # Micro-slippage / mid jump guard (if last_ticks available)
        last_ticks = md.get("last_ticks") or []
        if len(last_ticks) >= 3:
            mid_move = abs(last_ticks[-1] - last_ticks[-3]) / max(1e-9, last_ticks[-3]) * 1e4
            if mid_move > self.slippage_bps:
                return None

        # Wall proximity in direction of trade
        side = None
        if px > ma_f and ma_f > ma_s and imb > self.min_imbalance:
            side = "buy"
            if not nearest_wall_proximity(orderbook, px, "buy", depth=10, bps_thresh=5.0):
                return None
        elif px < ma_f and ma_f < ma_s and imb < -self.min_imbalance:
            side = "sell"
            if not nearest_wall_proximity(orderbook, px, "sell", depth=10, bps_thresh=5.0):
                return None

        if not side:
            return None

        conf = 0.65 + min(0.1, (abs(imb) - self.min_imbalance) * 0.2)
        return {
            "symbol": symbol,
            "side": side,
            "confidence": min(conf, 0.9),
            "metadata": {
                "type": "advanced_scalping",
                "spread": spread,
                "imbalance": imb,
                "ma_gap_bps": gap_bps
            }
        }
