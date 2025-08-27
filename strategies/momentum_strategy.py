from typing import Dict, Any, List
import numpy as np

class MomentumStrategy:
    """
    Lightweight momentum with HTF bias and ATR-sized exits.
    Emits proposals with entry=current, SL/TP from ATR bands.
    """
    def __init__(self, cfg, exchange):
        self.cfg = cfg
        self.exchange = exchange

    async def scan(self, market_data: Dict[str, Any], predictions: Dict[str, Any]) -> List[Dict]:
        out: List[Dict] = []
        for symbol in self.cfg.symbols_list():
            md = market_data.get(symbol) or {}
            ohlcv = md.get("ohlcv") or []
            if len(ohlcv) < 100: continue
            close = np.array([c[4] for c in ohlcv], dtype=float)
            high  = np.array([c[2] for c in ohlcv], dtype=float)
            low   = np.array([c[3] for c in ohlcv], dtype=float)

            # Simple ATR
            tr = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
            atr = tr[-14:].mean() if len(tr) >= 14 else tr.mean()
            px  = float(close[-1])

            # 50/200 cross as bias
            ma50 = close[-50:].mean() if len(close) >= 50 else close.mean()
            ma200 = close[-200:].mean() if len(close) >= 200 else close.mean()
            bias_up = ma50 > ma200
            bias_dn = ma50 < ma200

            # entry if small pullback in direction of bias
            if bias_up and px > ma50:
                sl = px - 1.25 * atr
                tp = px + 1.75 * atr
                out.append({"symbol":symbol,"side":"buy","entry":px,"stop":sl,"take":tp,
                            "confidence":0.62,"rr":(tp-px)/max(px-sl,1e-9),"edge_bps":3.0})
            elif bias_dn and px < ma50:
                sl = px + 1.25 * atr
                tp = px - 1.75 * atr
                out.append({"symbol":symbol,"side":"sell","entry":px,"stop":sl,"take":tp,
                            "confidence":0.62,"rr":(sl-px)/max(px-tp,1e-9),"edge_bps":3.0})
        return out
