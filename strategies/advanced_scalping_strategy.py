from typing import Dict, Any, List
import numpy as np

class AdvancedScalper:
    """
    Slightly stricter than SimpleScalper; checks micro MA slope as well.
    """
    def __init__(self, cfg, exchange):
        self.cfg = cfg
        self.exchange = exchange
        self.min_imb = float(getattr(cfg, "adv_min_imbalance", 0.10))
        self.gap_bps = float(getattr(cfg, "adv_ma_gap_bps", 5.0))

    async def scan(self, market_data: Dict[str, Any], predictions: Dict[str, Any]) -> List[Dict]:
        out: List[Dict] = []
        for symbol in self.cfg.symbols_list():
            md = market_data.get(symbol) or {}
            ohlcv = md.get("ohlcv") or []
            ob    = md.get("orderbook") or {}
            if len(ohlcv) < 80 or not ob.get("bids") or not ob.get("asks"):
                continue

            close = np.array([c[4] for c in ohlcv], dtype=float)
            px = float(close[-1])
            ma20 = close[-20:].mean()
            ma8  = close[-8:].mean()
            slope = ma8 - close[-16:-8].mean() if len(close) >= 24 else 0.0
            gap_bps = abs(px - ma20)/px*1e4

            best_bid = ob["bids"][0][0]; best_ask = ob["asks"][0][0]
            spread_bps = (best_ask - best_bid)/px*1e4
            bid_vol = sum([b[1] for b in ob["bids"][:5]]) if ob.get("bids") else 0.0
            ask_vol = sum([a[1] for a in ob["asks"][:5]]) if ob.get("asks") else 0.0
            imb = (bid_vol - ask_vol)/max(bid_vol + ask_vol, 1e-9)

            if spread_bps > 6:  # stricter
                continue

            if imb > self.min_imb and gap_bps > self.gap_bps and slope > 0:
                sl = px * (1 - 0.004)
                tp = px * (1 + 0.006)
                out.append({"symbol":symbol,"side":"buy","entry":px,"stop":sl,"take":tp,
                            "confidence":0.62,"rr":(tp-px)/max(px-sl,1e-9),"edge_bps":5.0,
                            "cost_bps":spread_bps+1.5})
            elif imb < -self.min_imb and gap_bps > self.gap_bps and slope < 0:
                sl = px * (1 + 0.004)
                tp = px * (1 - 0.006)
                out.append({"symbol":symbol,"side":"sell","entry":px,"stop":sl,"take":tp,
                            "confidence":0.62,"rr":(sl-px)/max(px-tp,1e-9),"edge_bps":5.0,
                            "cost_bps":spread_bps+1.5})
        return out
