from typing import Dict, Any, List
import numpy as np

class BTCEthPairs:
    """
    Z-score of log spread between BTC & ETH; market-neutral two-leg proposals.
    """
    def __init__(self, cfg, exchange):
        self.cfg = cfg; self.exchange = exchange
        self.lookback = 240
        self.entry_z  = 2.0
        self.exit_z   = 0.5

    async def scan(self, market_data: Dict[str, Any], predictions: Dict[str, Any]) -> List[Dict]:
        a = market_data.get("BTC/USDT:USDT", {}).get("ohlcv") or []
        b = market_data.get("ETH/USDT:USDT", {}).get("ohlcv") or []
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
        if z <= -self.entry_z:
            # long spread: long ETH, short BTC
            px_e, px_b = float(pb[-1]), float(pa[-1])
            atr_e = (pb[-1] - pb.max() if len(pb)>1 else 0.0)  # dummy ATR-ish
            atr_b = (pa.max() - pa[-1] if len(pa)>1 else 0.0)
            out += [
                {"symbol":"ETH/USDT:USDT","side":"buy","entry":px_e,"stop":px_e*0.996,"take":px_e*1.006,
                 "confidence":0.65,"rr":1.2,"edge_bps":2.0,"meta":{"pairs":True,"leg":"long_eth","beta":float(beta)}},
                {"symbol":"BTC/USDT:USDT","side":"sell","entry":px_b,"stop":px_b*1.004,"take":px_b*0.994,
                 "confidence":0.65,"rr":1.2,"edge_bps":2.0,"meta":{"pairs":True,"leg":"short_btc","beta":float(beta)}},
            ]
        elif z >= self.entry_z:
            px_e, px_b = float(pb[-1]), float(pa[-1])
            out += [
                {"symbol":"ETH/USDT:USDT","side":"sell","entry":px_e,"stop":px_e*1.004,"take":px_e*0.994,
                 "confidence":0.65,"rr":1.2,"edge_bps":2.0,"meta":{"pairs":True,"leg":"short_eth","beta":float(beta)}},
                {"symbol":"BTC/USDT:USDT","side":"buy","entry":px_b,"stop":px_b*0.996,"take":px_b*1.006,
                 "confidence":0.65,"rr":1.2,"edge_bps":2.0,"meta":{"pairs":True,"leg":"long_btc","beta":float(beta)}},
            ]
        return out
