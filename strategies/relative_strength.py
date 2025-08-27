from typing import Dict, Any, List
import numpy as np

class RelativeStrength:
    """
    Long strongest / short weakest among configured symbols over a lookback window.
    """
    def __init__(self, cfg, exchange):
        self.cfg = cfg; self.exchange = exchange
        self.lookback = 120
        self.cooldown_bars = 30
        self._last_bar = 0

    async def scan(self, market_data: Dict[str, Any], predictions: Dict[str, Any]) -> List[Dict]:
        bar_idx = market_data.get("_bar_index", 0)
        if bar_idx - self._last_bar < self.cooldown_bars:
            return []

        series = []
        for s in self.cfg.symbols_list():
            ohlcv = market_data.get(s, {}).get("ohlcv") or []
            if len(ohlcv) < self.lookback: return []
            px = np.array([x[4] for x in ohlcv], dtype=float)
            r = float(px[-1] / px[-self.lookback] - 1.0)
            series.append((s, r))
        series.sort(key=lambda x: x[1], reverse=True)
        long_sym, long_r = series[0]
        short_sym, short_r = series[-1]
        if long_sym == short_sym or (long_r - short_r) < 0.01:
            return []

        self._last_bar = bar_idx
        px_long = float(market_data[long_sym]["ohlcv"][-1][4])
        px_short = float(market_data[short_sym]["ohlcv"][-1][4])

        return [
            {"symbol":long_sym,"side":"buy","entry":px_long,"stop":px_long*0.996,"take":px_long*1.006,
             "confidence":0.6,"rr":1.2,"edge_bps":2.0,"meta":{"rs_gap": float(long_r-short_r)}},
            {"symbol":short_sym,"side":"sell","entry":px_short,"stop":px_short*1.004,"take":px_short*0.994,
             "confidence":0.6,"rr":1.2,"edge_bps":2.0,"meta":{"rs_gap": float(long_r-short_r)}},
        ]
