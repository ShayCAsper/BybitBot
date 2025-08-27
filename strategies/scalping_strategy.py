from typing import Dict, Any, List
import numpy as np

class SimpleScalper:
    """
    Very light microstructure scalper: uses spread + small MA gap + book imbalance.
    Emit only when costs are small (tight spread).
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
            if len(ohlcv) < 50: continue
            if not ob.get("bids") or not ob.get("asks"): continue

            close = np.array([c[4] for c in ohlcv], dtype=float)
            px = float(close[-1])
            ma20 = close[-20:].mean()
            gap_bps = abs(px - ma20)/px*1e4

            best_bid = ob["bids"][0][0]; best_ask = ob["asks"][0][0]
            spread_bps = (best_ask - best_bid)/px*1e4
            bid_vol = sum([b[1] for b in ob["bids"][:3]]) if ob.get("bids") else 0.0
            ask_vol = sum([a[1] for a in ob["asks"][:3]]) if ob.get("asks") else 0.0
            imb = (bid_vol - ask_vol)/max(bid_vol + ask_vol, 1e-9)

            if spread_bps > 8:  # too wide -> skip
                continue

            if imb > self.min_imb and gap_bps > self.gap_bps:
                # buy scalp
                sl = px * (1 - 0.005)
                tp = px * (1 + 0.005)
                out.append({"symbol":symbol,"side":"buy","entry":px,"stop":sl,"take":tp,
                            "confidence":0.6,"rr":(tp-px)/max(px-sl,1e-9),"edge_bps":4.0,
                            "cost_bps":spread_bps+2.0})
            elif imb < -self.min_imb and gap_bps > self.gap_bps:
                sl = px * (1 + 0.005)
                tp = px * (1 - 0.005)
                out.append({"symbol":symbol,"side":"sell","entry":px,"stop":sl,"take":tp,
                            "confidence":0.6,"rr":(sl-px)/max(px-tp,1e-9),"edge_bps":4.0,
                            "cost_bps":spread_bps+2.0})
        return out
