import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

class StrategyManager:
    """
    Coordinates all strategies, applies portfolio-level guardrails (regime, cooldown,
    ML confluence), and emits a compact list of high-quality signals.
    """

    def __init__(self, strategies_config: Dict[str, Any], exchange, risk_manager=None, ml_predictor=None):
        self.config = strategies_config or {}
        self.exchange = exchange
        self.risk_manager = risk_manager
        self.ml_predictor = ml_predictor

        from .scalping_strategy import ScalpingStrategy
        from .advanced_scalping_strategy import AdvancedScalpingStrategy
        from .momentum_strategy import MomentumStrategy
        from .mean_reversion_strategy import MeanReversionStrategy
        from .ml_strategy import MLStrategy

        self._cfg = lambda name, default={}: self.config.get(name, default)

        self.strategies = {
            "scalping": ScalpingStrategy(self._cfg("scalping"), self.exchange),
            "advanced_scalping": AdvancedScalpingStrategy(self._cfg("advanced_scalping"), self.exchange),
            "momentum": MomentumStrategy(self._cfg("momentum"), self.exchange),
            "mean_reversion": MeanReversionStrategy(self._cfg("mean_reversion"), self.exchange),
            "ml": MLStrategy(self._cfg("ml"), self.exchange, self.ml_predictor),
        }

        self.active_strategies: List[str] = self.config.get("active", ["scalping", "momentum", "mean_reversion"])
        self.symbol_cooldown = defaultdict(lambda: datetime.min)
        self.cooldown_secs = int(self.config.get("symbol_cooldown", 120))

        self.min_liquidity_usd = float(self.config.get("min_liquidity_usd", 2_000_000))
        self.min_vol_ratio = float(self.config.get("min_vol_ratio", 0.0005))

    async def generate_signals(self, market_data: Dict[str, Any], predictions: Optional[Dict] = None) -> List[Dict]:
        all_signals: List[Dict] = []
        for name in self.active_strategies:
            strat = self.strategies.get(name)
            if not strat:
                continue
            try:
                s_list = await strat.scan(market_data, predictions or {})
                if not s_list:
                    continue
                for s in s_list:
                    s.setdefault("strategy", name)
                    s.setdefault("weight", self._cfg(name, {}).get("weight", 1.0))
                    s.setdefault("confidence", s.get("confidence", 0.6))
                all_signals.extend(s_list)
            except Exception as e:
                import logging; logging.getLogger(__name__).error(f"Strategy '{name}' failed: {e}")

        if not all_signals:
            return []

        filtered_by_regime = [s for s in all_signals if self._regime_ok(market_data.get(s["symbol"], {}))]
        if not filtered_by_regime:
            return []

        filtered_by_ml = self._ml_confluence(filtered_by_regime, predictions or {})
        if not filtered_by_ml:
            return []

        cooled = [s for s in filtered_by_ml if self._cooldown_ok(s["symbol"])]
        if not cooled:
            return []

        return self._select_best_per_symbol(cooled)

    # ---------------- Guardrails ----------------
    def _cooldown_ok(self, symbol: str) -> bool:
        return datetime.now() >= self.symbol_cooldown[symbol]

    def notify_execution_result(self, symbol: str, success: bool):
        if not success:
            self.symbol_cooldown[symbol] = datetime.now() + timedelta(seconds=self.cooldown_secs)

    def _regime_ok(self, symbol_data: Dict[str, Any]) -> bool:
        if not symbol_data: return False
        ticker = symbol_data.get('ticker', {})
        ohlcv = symbol_data.get('ohlcv', [])
        if len(ohlcv) < 30: return False

        qv = float(ticker.get('quoteVolume', 0) or 0)
        if qv < self.min_liquidity_usd: return False

        highs = np.array([c[2] for c in ohlcv[-21:]])
        lows  = np.array([c[3] for c in ohlcv[-21:]])
        closes= np.array([c[4] for c in ohlcv[-21:]])
        tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
        atr = tr.mean() if len(tr) else 0.0
        px  = closes[-1] if len(closes) else 0.0
        if px <= 0: return False
        vol_ratio = atr / px
        return vol_ratio >= self.min_vol_ratio

    def _ml_confluence(self, signals: List[Dict], ml_predictions: Dict[str, Any]) -> List[Dict]:
        out = []
        for s in signals:
            sym = s["symbol"]
            mp = (ml_predictions or {}).get(sym)
            if not mp:
                out.append(s); continue
            ml_dir = 'buy' if mp.get('signal', 0) > 0 else 'sell' if mp.get('signal', 0) < 0 else None
            ml_conf = float(mp.get('confidence', 0.0))
            if ml_conf >= 0.80 and ml_dir:
                if ml_dir == s['side']:
                    s['confidence'] = min(float(s.get('confidence', 0.6)) * 1.10, 0.98)
                    out.append(s)
                else:
                    continue
            else:
                out.append(s)
        return out

    def _select_best_per_symbol(self, signals: List[Dict]) -> List[Dict]:
        best_by_symbol: Dict[str, Dict] = {}
        for s in signals:
            sym = s['symbol']
            score = float(s.get('confidence', 0.6)) * float(s.get('weight', 1.0))
            if sym not in best_by_symbol or score > best_by_symbol[sym].get('_score', -1):
                s['_score'] = score
                best_by_symbol[sym] = s
        for s in best_by_symbol.values():
            s.pop('_score', None)
        return list(best_by_symbol.values())
