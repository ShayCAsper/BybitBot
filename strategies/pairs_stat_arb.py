from typing import Dict, Any, List
import numpy as np

class BTCEthPairs:
    """
    Uses the first two symbols in the provided list as the pair (default: BTC & ETH).
    """
    def __init__(self, cfg, exchange, symbols: List[str]):
        self.cfg = cfg; self.exchange = exchange
        self.symbols = symbols
        self.lookback = 240
        self.entry_z  = 2.0
        self.exit_z   = 0.5

    async def scan(self, market_data: Dict[str, Any], predictions: Dict[str, Any]) -> List[Dict]:
        if len(self.symbols) < 2:
            return []
        a_sym, b_sym = self.symbols[0], self.symbols[1]
        a = market_data.get(a_sym, {}).get("ohlcv") or []
        b = market_data.get(b_sym, {}).get("ohlcv") or []
        if len(a) < self.lookback or len(b) < self.lookback: return []
        pa = np.array([x[4] for x in a][-self.lookback:], dtype=float)
        pb = np.array([x[4] for x in b][-self.lookback:], dtype=float)
        la, lb = np.log(pa), np.log(pb)
        X = np.vstack([la, np.ones_like(la)]).T
        beta, alpha = np.linalg.lstsq(X, lb, rcond=None)[0]
        spread = lb - (beta*la + alpha)
        mu, sd = spread.mean(), spread.std() + 1e-9
        z = float((spread[-1] - mu) / sd)

        out: List[Dict] = []
        px_a, px_b = float(pa[-1]), float(pb[-1])

        if z <= -self.entry_z:
            out += [
                {"symbol":b_sym,"side":"buy","entry":px_b,"stop":px_b*0.996,"take":px_b*1.006,
                 "confidence":0.65,"rr":1.2,"edge_bps":2.0,"meta":{"pairs":True,"leg":"long_B","beta":float(beta)}},
                {"symbol":a_sym,"side":"sell","entry":px_a,"stop":px_a*1.004,"take":px_a*0.994,
                 "confidence":0.65,"rr":1.2,"edge_bps":2.0,"meta":{"pairs":True,"leg":"short_A","beta":float(beta)}},
            ]
        elif z >= self.entry_z:
            out += [
                {"symbol":b_sym,"side":"sell","entry":px_b,"stop":px_b*1.004,"take":px_b*0.994,
                 "confidence":0.65,"rr":1.2,"edge_bps":2.0,"meta":{"pairs":True,"leg":"short_B","beta":float(beta)}},
                {"symbol":a_sym,"side":"buy","entry":px_a,"stop":px_a*0.996,"take":px_a*1.006,
                 "confidence":0.65,"rr":1.2,"edge_bps":2.0,"meta":{"pairs":True,"leg":"long_A","beta":float(beta)}},
            ]
        return out
