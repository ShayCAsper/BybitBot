from typing import Dict, Any, List, Optional
from loguru import logger
from core.master_router import Proposal

from strategies.momentum_strategy import MomentumStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.pairs_stat_arb import BTCEthPairs
from strategies.relative_strength import RelativeStrength
from strategies.scalping_strategy import SimpleScalper
from strategies.advanced_scalping_strategy import AdvancedScalper

class StrategyManager:
    def __init__(self, cfg, exchange_client):
        self.cfg = cfg
        self.exchange = exchange_client
        self.active_names = cfg.strategies()
        self.registry = self._build()

    def _build(self):
        reg = {}
        for name in self.active_names:
            try:
                symbols = self.cfg.strategy_symbols(name)
                if name == "momentum":
                    reg[name] = MomentumStrategy(self.cfg, self.exchange, symbols)
                elif name == "mean_reversion":
                    reg[name] = MeanReversionStrategy(self.cfg, self.exchange, symbols)
                elif name == "pairs":
                    reg[name] = BTCEthPairs(self.cfg, self.exchange, symbols)
                elif name == "rsr":
                    reg[name] = RelativeStrength(self.cfg, self.exchange, symbols)
                elif name == "scalping":
                    reg[name] = SimpleScalper(self.cfg, self.exchange, symbols)
                elif name == "advanced_scalping":
                    reg[name] = AdvancedScalper(self.cfg, self.exchange, symbols)
                logger.info(f"🧩 Strategy '{name}' symbols: {', '.join(symbols)}")
            except Exception as e:
                logger.error(f"Strategy init failed for {name}: {e}")
        return reg

    async def generate_proposals(self, market_data: Dict[str, Any], predictions: Optional[Dict[str, Any]] = None, only_strategies: Optional[List[str]] = None) -> List[Proposal]:
        names = self.active_names if not only_strategies else [n for n in only_strategies if n in self.active_names]
        proposals: List[Proposal] = []
        for name in names:
            strat = self.registry.get(name)
            if not strat:
                continue
            try:
                for s in await strat.scan(market_data, predictions or {}):
                    proposals.append(Proposal(
                        symbol=s["symbol"], side=s["side"], entry=s["entry"], stop=s["stop"], take=s["take"],
                        strategy=name, confidence=s.get("confidence", 0.6),
                        exp_edge_bps=s.get("edge_bps", 2.0), exp_rr=s.get("rr", 1.0),
                        cost_bps=s.get("cost_bps", 2.0), qty=s.get("quantity"), meta=s.get("meta")
                    ))
            except Exception as e:
                logger.error(f"Strategy '{name}' failed: {e}")
        return proposals
