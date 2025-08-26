from typing import Dict, List, Any, Optional
from .integrate_scalping import compute_spread_and_mid, orderbook_imbalance

class ScalpingStrategy:
    """
    Baseline scalper: spread + imbalance + quick anti-widening check.
    """

    def __init__(self, config: Dict[str, Any], exchange):
        self.config = config or {}
        self.exchange = exchange

        self.max_spread = float(self.config.get("max_spread", 0.0008))  # 8 bps
        self.min_imbalance = float(self.config.get("min_imbalance", 0.15))

    async def scan(self, market_data: Dict[str, Any], predictions: Dict[str, Any]) -> List[Dict]:
        out: List[Dict] = []
        for symbol, md in market_data.items():
            s = await self._analyze_symbol(symbol, md)
            if s:
                out.append(s)
        return out

    async def _analyze_symbol(self, symbol: str, md: Dict[str, Any]) -> Optional[Dict]:
        orderbook = md.get("orderbook") or {}
        if not orderbook:
            return None

        spread, bid, ask = compute_spread_and_mid(orderbook)
        if spread <= 0 or spread > self.max_spread:
            return None

        imb = orderbook_imbalance(orderbook, depth=5)

        # Side by imbalance
        side = "buy" if imb > self.min_imbalance else "sell" if imb < -self.min_imbalance else None
        if not side:
            return None

        # Re-check spread right before emit (anti-widening)
        re_spread, _, _ = compute_spread_and_mid(orderbook)
        if re_spread > spread * 1.5:
            return None

        return {
            "symbol": symbol,
            "side": side,
            "confidence": 0.6,
            "weight": 0.9,
            "metadata": {"type": "scalping", "spread": spread, "imbalance": imb}
        }
