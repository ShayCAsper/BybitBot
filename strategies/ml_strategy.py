from typing import Dict, List, Any

class MLStrategy:
    """
    Thin wrapper to turn ML model outputs into optional signals.
    Also provides predictions to StrategyManager for confluence.
    """

    def __init__(self, config: Dict[str, Any], exchange, ml_predictor=None):
        self.config = config or {}
        self.exchange = exchange
        self.ml_predictor = ml_predictor

        # Only emit standalone ML signals when confidence is very high
        self.min_conf_for_signal = float(self.config.get("min_conf_for_signal", 0.9))

    async def scan(self, market_data: Dict[str, Any], predictions: Dict[str, Any]) -> List[Dict]:
        """
        If ML is present and very confident, emit direct signals too.
        (StrategyManager will also use ML as a veto/boost step.)
        Expected prediction format per symbol: {"signal": +1 | -1 | 0, "confidence": 0..1}
        """
        out: List[Dict] = []
        preds = predictions or {}
        if not preds and self.ml_predictor:
            try:
                preds = await self.ml_predictor.predict(market_data)
            except Exception:
                preds = {}

        for symbol, mp in (preds or {}).items():
            conf = float(mp.get("confidence", 0.0))
            sigv = mp.get("signal", 0)
            if conf >= self.min_conf_for_signal and sigv != 0:
                out.append({
                    "symbol": symbol,
                    "side": "buy" if sigv > 0 else "sell",
                    "confidence": min(0.7 + (conf - self.min_conf_for_signal) * 0.2, 0.95),
                    "weight": 1.1,
                    "metadata": {"type": "ml", "ml_confidence": conf}
                })
        return out
