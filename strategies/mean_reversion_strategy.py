from typing import Dict, List, Any, Optional
import numpy as np

def _sma(arr, n):
    if len(arr) < n:
        return float(np.mean(arr)) if len(arr) else 0.0
    return float(np.mean(arr[-n:]))

def _stdev(arr, n):
    if len(arr) < n:
        return float(np.std(arr)) if len(arr) else 0.0
    return float(np.std(arr[-n:]))

def _rsi(closes: np.ndarray, period: int = 14) -> float:
    diffs = np.diff(closes)
    gains = np.clip(diffs, 0, None)
    losses = -np.clip(diffs, None, 0)
    avg_gain = gains[-period:].mean() if len(gains) >= period else gains.mean() if len(gains) else 0
    avg_loss = losses[-period:].mean() if len(losses) >= period else losses.mean() if len(losses) else 1e-9
    rs = avg_gain / max(avg_loss, 1e-9)
    return 100 - (100 / (1 + rs))

class MeanReversionStrategy:
    """
    Bollinger / z-score / RSI based entries with hysteresis and time-stop metadata.
    """

    def __init__(self, config: Dict[str, Any], exchange):
        self.config = config or {}
        self.exchange = exchange

        self.bb_len = int(self.config.get("bb_len", 20))
        self.std_threshold = float(self.config.get("std_threshold", 2.0))
        self.hysteresis = float(self.config.get("hysteresis", 0.5))
        self.time_stop = int(self.config.get("time_stop", 20))  # bars

    async def scan(self, market_data: Dict[str, Any], predictions: Dict[str, Any]) -> List[Dict]:
        out: List[Dict] = []
        for symbol, md in market_data.items():
            s = await self._analyze_symbol(symbol, md)
            if s:
                out.append(s)
        return out

    async def _analyze_symbol(self, symbol: str, md: Dict[str, Any]) -> Optional[Dict]:
        ohlcv = md.get("ohlcv") or []
        if len(ohlcv) < self.bb_len + 5:
            return None

        closes = np.array([c[4] for c in ohlcv], dtype=float)
        px = closes[-1]

        ma = _sma(closes, self.bb_len)
        sd = _stdev(closes, self.bb_len)
        if sd <= 0:
            return None

        bb_upper = ma + 2 * sd
        bb_lower = ma - 2 * sd
        z = (px - ma) / sd
        rsi = _rsi(closes, 14)

        # Hysteresis: require "hook" back by 0.5σ vs threshold
        buy_ok = (z < -self.std_threshold + self.hysteresis) and (rsi < 30) and (px < bb_lower)
        sell_ok = (z >  self.std_threshold - self.hysteresis) and (rsi > 70) and (px > bb_upper)

        if buy_ok:
            return {
                "symbol": symbol,
                "side": "buy",
                "confidence": 0.6,
                "weight": 0.9,
                "metadata": {"type": "mean_reversion", "time_stop_bars": self.time_stop, "zscore": float(z), "rsi": float(rsi)}
            }
        if sell_ok:
            return {
                "symbol": symbol,
                "side": "sell",
                "confidence": 0.6,
                "weight": 0.9,
                "metadata": {"type": "mean_reversion", "time_stop_bars": self.time_stop, "zscore": float(z), "rsi": float(rsi)}
            }
        return None
