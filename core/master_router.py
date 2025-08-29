from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import math, time

@dataclass
class Proposal:
    symbol: str
    side: str               # "buy" | "sell"
    entry: float
    stop: float
    take: float
    strategy: str
    confidence: float = 0.6
    exp_edge_bps: float = 0.0   # expected alpha net of costs
    exp_rr: float = 1.0
    cost_bps: float = 2.0       # fees+slippage estimate
    qty: Optional[float] = None
    meta: Optional[Dict] = None

    def score(self) -> float:
        rr_term = max(self.exp_rr, 0.01)
        edge_term = (self.exp_edge_bps - self.cost_bps) / 10.0
        conf_term = self.confidence
        return math.tanh(0.6 * (edge_term + conf_term + 0.15 * math.log1p(rr_term)))

class EnsembleAllocator:
    """
    Rolling per-strategy 'quality' via EW mean/variance, then choose best proposals
    within a risk budget.
    """
    def __init__(self, half_life_trades: int = 50):
        self.stats: Dict[str, Dict[str, float]] = {}
        self.half_life = max(10, half_life_trades)

    def update_with_fill(self, strategy: str, pnl_usd: float):
        st = self.stats.setdefault(strategy, {"p":0.0, "v":1e-6, "n":0.0})
        alpha = 2.0 / (self.half_life + 1.0)
        old_p, old_v = st["p"], st["v"]
        st["p"] = (1 - alpha) * old_p + alpha * pnl_usd
        st["v"] = (1 - alpha) * old_v + alpha * (pnl_usd - st["p"])**2
        st["n"] = min(st["n"] + 1.0, 10 * self.half_life)

    def _score_strategy(self, name: str) -> float:
        st = self.stats.get(name, {"p":0.0, "v":1e-6, "n":0.0})
        mu, var = st["p"], max(st["v"], 1e-6)
        sharpish = mu / math.sqrt(var)
        bonus = 0.15 / math.sqrt(st["n"] + 1.0)
        return sharpish + bonus

    def choose(self, proposals: List[Proposal], risk_budget_usd: float, max_concurrent: int = 2) -> List[Proposal]:
        if not proposals:
            return []
        weighted = []
        for p in proposals:
            strat_q = self._score_strategy(p.strategy)
            weighted.append(((1.0 + strat_q) * p.score(), p))
        weighted.sort(key=lambda x: x[0], reverse=True)

        chosen: List[Proposal] = []
        used = 0.0
        for _, p in weighted:
            if p.qty is None and p.entry and p.stop:
                risk_per_trade = max(0.002 * risk_budget_usd, 5.0)  # 0.2% equity
                size_usd = risk_per_trade / max(abs(p.entry - p.stop), 1e-9)
                p.qty = round(size_usd, 6)
            chosen.append(p)
            used += (p.qty or 0.0) * p.entry
            if len(chosen) >= max_concurrent or used >= risk_budget_usd:
                break
        return chosen
