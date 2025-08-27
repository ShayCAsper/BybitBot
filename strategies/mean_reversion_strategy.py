from typing import Dict, Any, List
import numpy as np

class MeanReversionStrategy:
    def __init__(self, cfg, exchange, symbols: List[str]):
        self.cfg = cfg
        self.exchange = exchange
        self.symbols = symbols
        self.bb_len = 20

    async def scan(self, market_data: Dict[str, Any], predictions: Dict[str, Any]) -> List[Dict]:
        out: List[Dict] = []
        for symbol in self.symbols:
            md = market_data.get(symbol) or {}
            ohlcv = md.get("ohlcv") or []
            if len(ohlcv) < self.bb_len + 5: continue
            close = np.array([c[4] for c in ohlcv], dtype=float)
            px = float(close[-1])
            ma = close[-self.bb_len:].mean()
            sd = close[-self.bb_len:].std() + 1e-9
            upper, lower = ma + 2*sd, ma - 2*sd
            z = (px - ma)/sd

            diffs = np.diff(close)
            gains = np.clip(diffs, 0, None); losses = -np.clip(diffs, None, 0)
            avg_g = gains[-14:].mean() if len(gains)>=14 else gains.mean() if len(gains) else 0.0
            avg_l = losses[-14:].mean() if len(losses)>=14 else losses.mean() if len(losses) else 1e-9
            rsi = 100 - 100/(1+(avg_g/max(avg_l,1e-9)))

            if z < -2.0 and rsi < 30 and px < lower:
                sl = px - 1.5*sd
                tp = ma
                out.append({"symbol":symbol,"side":"buy","entry":px,"stop":sl,"take":tp,
                            "confidence":0.6,"rr":(tp-px)/max(px-sl,1e-9),"edge_bps":2.0,
                            "meta":{"time_stop_bars":20}})
            elif z > 2.0 and rsi > 70 and px > upper:
                sl = px + 1.5*sd
                tp = ma
                out.append({"symbol":symbol,"side":"sell","entry":px,"stop":sl,"take":tp,
                            "confidence":0.6,"rr":(sl-px)/max(px-tp,1e-9),"edge_bps":2.0,
                            "meta":{"time_stop_bars":20}})
        return out
