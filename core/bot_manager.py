import os, asyncio, time
from typing import Dict, Any, List, Optional
from loguru import logger
from datetime import datetime, timezone
from core.master_router import Proposal, EnsembleAllocator

class BotManager:
    def __init__(self, config):
        self.cfg = config
        self.exchange = None

        # State
        self.active = False
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.balance = 0.0
        self.initial_balance = 0.0

        # Performance / risk
        r = self.cfg.risk()
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.daily_loss_cap_pct = r["daily_loss_cap_pct"]
        self.max_daily_trades   = r["max_daily_trades"]
        self.max_positions      = r["max_positions"]
        self.min_rr             = r["min_rr"]
        self.max_consec_losses  = r["max_consec_losses"]
        self.loss_cooldown_min  = r["loss_cooldown_min"]
        self.session_hours_utc  = r["session_hours_utc"]

        # Loop cadence
        self.base_interval = self.cfg.base_interval
        self.current_interval = self.base_interval
        self.loop_log_every = self.cfg.loop_log_every
        self.loop_counter = 0

        # Strategy cadence
        self.strategy_intervals = self.cfg.cadence()
        self.strategy_next_run  = {k: 0.0 for k in self.strategy_intervals}

        # Allocator
        self.allocator = EnsembleAllocator(half_life_trades=50)

        # Strategy manager (created in startup)
        self.strategy_manager = None

    async def startup(self, exchange):
        self.exchange = exchange
        if self.cfg.force_one_way:
            await self.exchange.set_position_mode(one_way=True)

        # load balance if available
        try:
            bal = await self.exchange.exchange.fetch_balance()
            self.balance = float(bal.get("USDT", {}).get("total") or bal.get("USDT", {}).get("free") or 0.0)
        except Exception:
            self.balance = float(os.getenv("START_BALANCE", "10000"))
        self.initial_balance = self.balance or float(os.getenv("START_BALANCE", "10000"))

        logger.info("⚙️ Trade Control Settings:")
        logger.info(f"  Max positions total: {self.max_positions}")
        logger.info(f"  Max daily trades: {self.max_daily_trades}")
        logger.info(f"  Daily loss cap: -{self.daily_loss_cap_pct}%")
        logger.info(f"  Strategy intervals: {self.strategy_intervals}")

        # Create strategy manager
        from strategies.strategy_manager import StrategyManager
        self.strategy_manager = StrategyManager(self.cfg, self.exchange)

    async def start(self):
        self.active = True
        logger.info(f"🚀 Bot Started! Trading: ENABLED ✅")
        try:
            while self.active:
                await self.trading_loop()
        finally:
            await self.exchange.close()

    async def stop(self):
        self.active = False

    def _in_session(self) -> bool:
        try:
            start_h, end_h = [int(x) for x in self.session_hours_utc.split("-")]
        except Exception:
            return True
        now_h = datetime.now(timezone.utc).hour
        if start_h <= end_h:
            return start_h <= now_h < end_h
        return now_h >= start_h or now_h < end_h

    def _calc_rr(self, entry: float, sl: float, tp: float, side: str) -> float:
        if not entry or not sl or not tp: return 0.0
        if side.lower() == "buy":
            risk = max(entry - sl, 1e-9); reward = max(tp - entry, 0.0)
        else:
            risk = max(sl - entry, 1e-9); reward = max(entry - tp, 0.0)
        return reward / risk if risk > 0 else 0.0

    async def fetch_market_snapshot(self) -> Dict[str, Any]:
        """
        Pull small snapshots per symbol (OHLCV + top-of-book).
        """
        out: Dict[str, Any] = {"_bar_index": int(time.time() // 60)}
        for s in self.cfg.symbols_list():
            try:
                ohlcv = await self.exchange.exchange.fetch_ohlcv(s, timeframe="1m", limit=240)
                ob    = await self.exchange.fetch_order_book(s, limit=5)
                out[s] = {"ohlcv": ohlcv, "orderbook": ob}
            except Exception as e:
                logger.warning(f"snapshot failed for {s}: {e}")
        return out

    async def trading_loop(self):
        self.loop_counter += 1
        if self.loop_counter % max(1, self.loop_log_every) == 0:
            logger.info(f"🔄 Trading Loop #{self.loop_counter} (Interval: {self.current_interval}s)")

        # Session & global brakes
        if not self._in_session():
            await asyncio.sleep(self.current_interval); return
        if self.daily_trades >= self.max_daily_trades:
            logger.warning("Daily trade cap reached. Pausing.")
            await asyncio.sleep(max(30, self.current_interval)); return

        now = time.monotonic()
        due = []
        for name in self.cfg.strategies():
            interval = self.strategy_intervals.get(name, self.base_interval)
            nxt = self.strategy_next_run.get(name, 0.0)
            if now >= nxt:
                due.append(name)
        if not due:
            await asyncio.sleep(self.current_interval); return

        # 1) Snapshot
        market = await self.fetch_market_snapshot()

        # 2) Predictions cadence (optional)
        predictions = {}

        # 3) Generate signals only for due strategies
        proposals: List[Proposal] = await self.strategy_manager.generate_proposals(
            market_data=market, predictions=predictions, only_strategies=due
        )

        # 4) Update next-run
        for name in due:
            self.strategy_next_run[name] = now + max(1, self.strategy_intervals.get(name, self.base_interval))

        # 5) Score/allocate
        risk_budget = max(0.05 * (self.balance or self.initial_balance), 200.0)  # 5% of equity or $200
        chosen = self.allocator.choose(proposals, risk_budget_usd=risk_budget, max_concurrent=2)
        logger.info(f"📊 {len(chosen)} chosen of {len(proposals)} proposals")

        # 6) Execute
        for p in chosen:
            rr = self._calc_rr(p.entry, p.stop, p.take, p.side)
            # RR floors per strategy
            floor = self.min_rr.get(p.strategy, 1.0)
            if rr < floor:
                logger.info(f"⛔ Skip {p.symbol} {p.side} [{p.strategy}] RR {rr:.2f} < {floor}")
                continue

            qty = p.qty or 0.0
            order = await self.exchange.place_order_with_sl_tp(
                symbol=p.symbol, side=p.side, quantity=qty,
                price=None, order_type="market",  # or pass limit if your strategy provides
                stop_loss=p.stop, take_profit=p.take, reduce_only=False,
                signal_strategy=p.strategy
            )
            if order:
                self.daily_trades += 1
                logger.info(f"✅ Placed {p.strategy} {p.side.upper()} {p.symbol} qty={qty}")
            else:
                logger.warning(f"Order failed for {p.symbol}")

        await asyncio.sleep(self.current_interval)
