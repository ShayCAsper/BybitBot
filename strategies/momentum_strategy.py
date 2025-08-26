from typing import Dict, List, Any, Optional
import numpy as np

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    k = 2 / (period + 1)
    ema = np.empty_like(arr, dtype=float)
    ema[0] = arr[0]
    for i in range(1, len(arr)):
        ema[i] = arr[i] * k + ema[i-1] * (1 - k)
    return ema

def _rsi(closes: np.ndarray, period: int = 14) -> float:
    diffs = np.diff(closes)
    gains = np.clip(diffs, 0, None)
    losses = -np.clip(diffs, None, 0)
    avg_gain = gains[-period:].mean() if len(gains) >= period else gains.mean() if len(gains) else 0
    avg_loss = losses[-period:].mean() if len(losses) >= period else losses.mean() if len(losses) else 1e-9
    rs = avg_gain / max(avg_loss, 1e-9)
    return 100 - (100 / (1 + rs))

class MomentumStrategy:
    """
    Trend-following momentum with HTF bias and ATR-scaled exits.
    """

    def __init__(self, config: Dict[str, Any], exchange):
        self.config = config or {}
        self.exchange = exchange

        self.fast = int(self.config.get("macd_fast", 12))
        self.slow = int(self.config.get("macd_slow", 26))
        self.signal = int(self.config.get("macd_signal", 9))
        self.rsi_thr_buy = float(self.config.get("rsi_thr_buy", 55))
        self.rsi_thr_sell = float(self.config.get("rsi_thr_sell", 45))

        self.htf_period = int(self.config.get("htf_period", 200))  # "200 on 5m"
        self.exit_atr_mult = float(self.config.get("exit_atr_mult", 1.2))

    async def scan(self, market_data: Dict[str, Any], predictions: Dict[str, Any]) -> List[Dict]:
        out: List[Dict] = []
        for symbol, md in market_data.items():
            s = await self._analyze_symbol(symbol, md)
            if s:
                out.append(s)
        return out

    async def _analyze_symbol(self, symbol: str, md: Dict[str, Any]) -> Optional[Dict]:
        ohlcv = md.get("ohlcv") or []
        if len(ohlcv) < max(60, self.htf_period + 5):
            return None
        closes = np.array([c[4] for c in ohlcv], dtype=float)

        # MACD
        ema_fast = _ema(closes, self.fast)
        ema_slow = _ema(closes, self.slow)
        macd = ema_fast - ema_slow
        macd_signal = _ema(macd, self.signal)
        macd_hist = macd - macd_signal

        # RSI
        rsi = _rsi(closes, 14)

        px = closes[-1]

        # HTF bias (approximate 5m downsample: every 5 bars if your base TF is 1m)
        closes_5 = closes[::5] if len(closes) >= 5 else closes
        if len(closes_5) < self.htf_period:
            htf_ma = closes_5.mean()
        else:
            htf_ma = closes_5[-self.htf_period:].mean()

        long_bias = px > htf_ma
        short_bias = px < htf_ma

        signal = None
        if macd_hist[-1] > 0 and rsi >= self.rsi_thr_buy and long_bias:
            side = "buy"
            signal = {"symbol": symbol, "side": side}
        elif macd_hist[-1] < 0 and rsi <= self.rsi_thr_sell and short_bias:
            side = "sell"
            signal = {"symbol": symbol, "side": side}
        else:
            return None

        # ATR for exits
        highs = np.array([c[2] for c in ohlcv[-15:]], dtype=float)
        lows  = np.array([c[3] for c in ohlcv[-15:]], dtype=float)
        prev_close = np.array([c[4] for c in ohlcv[-16:-1]], dtype=float)
        tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
        atr = tr.mean() if len(tr) else max(px * 0.002, 1.0)

        if signal["side"] == "buy":
            sl = px - self.exit_atr_mult * atr
            tp = px + 1.5 * self.exit_atr_mult * atr
        else:
            sl = px + self.exit_atr_mult * atr
            tp = px - 1.5 * self.exit_atr_mult * atr

        signal.update({
            "confidence": 0.65,
            "weight": 1.0,
            "stop_loss": float(sl),
            "take_profit": float(tp),
            "metadata": {"type": "momentum", "rsi": float(rsi), "macd_hist": float(macd_hist[-1])}
        })
        return signal
