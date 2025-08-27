import os, asyncio, time
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from loguru import logger
from core.master_router import Proposal, EnsembleAllocator

def _env_float(key: str, default: float) -> float:
    try: return float(os.getenv(key, default))
    except Exception: return default

def _env_int(key: str, default: int) -> int:
    try: return int(float(os.getenv(key, default)))
    except Exception: return default

class BotManager:
    def __init__(self, config):
        self.cfg = config
        self.exchange = None

        # State
        self.active = False
        # positions keyed by symbol
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.balance = 0.0
        self.initial_balance = 0.0

        # Risk / performance knobs
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

        # Cooldowns (per strategy)
        self.cooldowns = {
            "advanced_scalping": _env_int("COOLDOWN_ADV_SCALP", 60),
            "scalping":          _env_int("COOLDOWN_SCALP", 60),
            "momentum":          _env_int("COOLDOWN_MOMENTUM", 180),
            "mean_reversion":    _env_int("COOLDOWN_MEANREV", 300),
            "pairs":             _env_int("COOLDOWN_PAIRS", 120),
            "rsr":               _env_int("COOLDOWN_RSR", 120),
        }
        self.last_trade_ts: Dict[str, float] = {}

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

        # Performance HUD
        self.monitor_every = _env_int("MONITOR_EVERY_SEC", 60)
        self._monitor_task: Optional[asyncio.Task] = None

        # Trailing manager
        self.trailing_cfg = self.cfg.trailing
        self.trailing_enabled = bool(getattr(self.trailing_cfg, "enabled", True))
        self.trail_poll_sec = _env_int("TRAIL_POLL_SEC", 2)
        self._trail_task: Optional[asyncio.Task] = None

        # Market microstructure guards
        self.max_spread_bps_scalp   = _env_float("MAX_SPREAD_BPS_SCALP", 8.0)
        self.max_spread_bps_default = _env_float("MAX_SPREAD_BPS_DEFAULT", 12.0)
        self.min_depth_usd          = _env_float("MIN_DEPTH_USD", 50_000.0)

    async def startup(self, exchange):
        self.exchange = exchange
        if self.cfg.force_one_way:
            await self.exchange.set_position_mode(one_way=True)

        # Balance
        try:
            bal = await self.exchange.exchange.fetch_balance()
            self.balance = float(bal.get("USDT", {}).get("total") or bal.get("USDT", {}).get("free") or 0.0)
        except Exception as e:
            logger.warning(f"fetch_balance on startup failed: {e}")
            self.balance = float(os.getenv("START_BALANCE", "10000"))
        self.initial_balance = self.balance or float(os.getenv("START_BALANCE", "10000"))

        logger.info("⚙️ Trade Control Settings:")
        logger.info(f"  Max positions total: {self.max_positions}")
        logger.info(f"  Max daily trades: {self.max_daily_trades}")
        logger.info(f"  Daily loss cap: -{self.daily_loss_cap_pct}%")
        logger.info(f"  Strategy intervals: {self.strategy_intervals}")
        logger.info(f"  Trailing: enabled={self.trailing_enabled}, mode={self.trailing_cfg.mode}, "
                    f"percent={self.trailing_cfg.percent}%, breakeven_at={self.trailing_cfg.breakeven_at}%")

        # Create strategy manager
        from strategies.strategy_manager import StrategyManager
        self.strategy_manager = StrategyManager(self.cfg, self.exchange)

    async def start(self):
        self.active = True
        logger.info("🚀 Bot Started! Trading: ENABLED ✅")
        if not self._monitor_task:
            self._monitor_task = asyncio.create_task(self.performance_monitor_loop())
        if self.trailing_enabled and not self._trail_task:
            self._trail_task = asyncio.create_task(self.trailing_manager_loop())
        try:
            while self.active:
                await self.trading_loop()
        finally:
            if self._monitor_task:
                self._monitor_task.cancel()
                self._monitor_task = None
            if self._trail_task:
                self._trail_task.cancel()
                self._trail_task = None
            await self.exchange.close()

    async def stop(self):
        self.active = False

    # ---------- Performance Monitor ----------
    async def performance_monitor_loop(self):
        while self.active:
            try:
                await self.performance_monitor()
            except Exception as e:
                logger.warning(f"performance_monitor error: {e}")
            await asyncio.sleep(max(10, self.monitor_every))

    async def performance_monitor(self):
        try:
            bal = await self.exchange.exchange.fetch_balance()
            usdt = bal.get("USDT", {}).get("total") or bal.get("USDT", {}).get("free")
            if usdt is not None:
                self.balance = float(usdt)
        except Exception:
            pass

        init = self.initial_balance or self.balance or 0.0
        cur  = self.balance or init
        pnl_usd = cur - init
        pnl_pct = (pnl_usd / init * 100.0) if init else 0.0

        # open positions (approx)
        open_pos = 0
        try:
            poss = await self.exchange.exchange.fetch_positions(params={"category": "linear"})
            for p in poss or []:
                info = p.get("info") or {}
                size = 0.0
                for k in ("contracts","contractSize","positionAmt"):
                    v = p.get(k)
                    if v is not None:
                        try: size = float(v); break
                        except Exception: pass
                if not size:
                    for k in ("size","positionValue","positionAmt"):
                        v = info.get(k)
                        if v is not None:
                            try: size = float(v); break
                            except Exception: pass
                if abs(size) > 0: open_pos += 1
        except Exception:
            open_pos = len(self.positions or {})

        now = time.time()
        recent_1h = sum(1 for t in (self.trade_history or []) if (now - t.get("ts", now)) <= 3600)

        logger.info("📊 === PERFORMANCE UPDATE ===")
        logger.info(f"Balance: ${cur:,.2f}")
        sign = "+" if pnl_usd >= 0 else ""
        logger.info(f"P&L: {sign}${pnl_usd:,.2f} ({sign}{pnl_pct:.2f}%)")
        logger.info(f"Open Positions: {open_pos}/{self.max_positions}")
        logger.info(f"Daily Trades: {self.daily_trades}")
        logger.info(f"Recent Trades (1h): {recent_1h}")
        logger.info(f"Current Interval: {self.current_interval}s")

    # ---------- Trailing Manager ----------
    async def trailing_manager_loop(self):
        """
        Background loop that:
          - sets breakeven once profit >= breakeven_at
          - then trails stop by trailing.percent
          - removes local position state if Bybit position is closed
        """
        while self.active and self.trailing_enabled:
            try:
                await self._manage_all_trails()
            except Exception as e:
                logger.warning(f"trailing_manager error: {e}")
            await asyncio.sleep(max(1, self.trail_poll_sec))

    async def _manage_all_trails(self):
        # copy keys to avoid runtime dict change issues
        for sym in list(self.positions.keys()):
            pos = self.positions.get(sym)
            if not pos: continue

            # if exchange shows no open pos, drop our local record
            still_open = await self.exchange.has_open_position(sym)
            if not still_open:
                logger.info(f"📉 Position closed on exchange: {sym}. Removing from local state.")
                self.positions.pop(sym, None)
                continue

            price = await self.exchange.get_price(sym)
            if price is None: continue

            side = pos["side"].lower()
            entry = float(pos["entry"])
            cur_sl = pos.get("current_sl")
            tp = pos.get("tp")
            trail_pct = float(self.trailing_cfg.percent)
            breakeven_at = float(self.trailing_cfg.breakeven_at)

            # initialize peak/trough
            if side == "buy":
                pos["peak"] = max(float(pos.get("peak", entry)), price)
                # set breakeven?
                up_pct = (pos["peak"] - entry) / entry * 100.0
                if not pos.get("breakeven_set", False) and up_pct >= breakeven_at:
                    new_sl = max(float(pos.get("sl") or 0.0), entry)
                    if cur_sl is None or new_sl > cur_sl:
                        ok = await self.exchange.set_stop_loss(sym, new_sl)
                        if ok:
                            pos["current_sl"] = new_sl
                            pos["breakeven_set"] = True
                            logger.info(f"🟢 Breakeven set for {sym} @ {new_sl:.4f}")

                # trail
                trail_stop = pos["peak"] * (1 - trail_pct / 100.0)
                # don't trail beyond TP
                if tp: trail_stop = min(trail_stop, float(tp))
                if (cur_sl is None or trail_stop > cur_sl) and trail_stop < price:
                    ok = await self.exchange.set_stop_loss(sym, trail_stop)
                    if ok:
                        pos["current_sl"] = trail_stop
                        logger.info(f"🔧 Trailing SL moved up for {sym} -> {trail_stop:.4f}")

            else:  # short
                pos["trough"] = min(float(pos.get("trough", entry)), price)
                down_pct = (entry - pos["trough"]) / entry * 100.0
                if not pos.get("breakeven_set", False) and down_pct >= breakeven_at:
                    new_sl = min(float(pos.get("sl") or 1e18), entry)
                    if cur_sl is None or new_sl < cur_sl:
                        ok = await self.exchange.set_stop_loss(sym, new_sl)
                        if ok:
                            pos["current_sl"] = new_sl
                            pos["breakeven_set"] = True
                            logger.info(f"🟢 Breakeven set for {sym} (SHORT) @ {new_sl:.4f}")

                trail_stop = pos["trough"] * (1 + trail_pct / 100.0)
                if tp: trail_stop = max(trail_stop, float(tp))
                if (cur_sl is None or trail_stop < cur_sl) and trail_stop > price:
                    ok = await self.exchange.set_stop_loss(sym, trail_stop)
                    if ok:
                        pos["current_sl"] = trail_stop
                        logger.info(f"🔧 Trailing SL moved down for {sym} (SHORT) -> {trail_stop:.4f}")

    # ---------- Helpers ----------
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

    async def _pre_trade_guard(self, p: Proposal) -> bool:
        # daily loss cap
        if self.initial_balance > 0 and self.balance > 0:
            dd_pct = (self.initial_balance - self.balance) / self.initial_balance * 100.0
            if dd_pct >= self.daily_loss_cap_pct:
                logger.error(f"⛔ Daily loss cap hit ({dd_pct:.2f}% ≥ {self.daily_loss_cap_pct}%). Pausing entries.")
                return False

        # daily trade cap
        if self.daily_trades >= self.max_daily_trades:
            logger.warning("⏸ Daily trade cap reached.")
            return False

        # max positions
        if len(self.positions) >= self.max_positions:
            logger.warning("⏸ Max positions open. Skipping new entry.")
            return False

        # per-strategy cooldown
        lt = self.last_trade_ts.get(p.strategy, 0.0)
        cd = self.cooldowns.get(p.strategy, 120)
        if time.time() - lt < cd:
            return False

        # RR floor
        rr = self._calc_rr(p.entry, p.stop, p.take, p.side)
        floor = self.min_rr.get(p.strategy, 1.0)
        if rr < floor:
            logger.info(f"⛔ Skip {p.symbol} {p.side} [{p.strategy}] RR {rr:.2f} < {floor}")
            return False

        # microstructure: spread/depth
        spread_bps, bid_depth, ask_depth = await self.exchange.get_spread_metrics(p.symbol)
        max_spread = self.max_spread_bps_scalp if p.strategy in ("scalping","advanced_scalping") else self.max_spread_bps_default
        depth_ok = (bid_depth >= self.min_depth_usd and ask_depth >= self.min_depth_usd)
        if spread_bps > max_spread or not depth_ok:
            logger.info(
                f"⛔ Microstructure guard: spread={spread_bps:.1f}bps (max {max_spread}), "
                f"bid_depth=${bid_depth:,.0f}, ask_depth=${ask_depth:,.0f} (min ${self.min_depth_usd:,.0f})"
            )
            return False

        return True

   
    # ---------- Main loop ----------
    async def fetch_market_snapshot(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"_bar_index": int(time.time() // 60)}
        markets = (self.exchange.exchange.markets or {})
        for s in self.cfg.symbols_list():
            if s not in markets:
                # silently skip unknown/unsupported symbols on this venue/testnet
                continue
            try:
                ohlcv = await self.exchange.exchange.fetch_ohlcv(s, timeframe="1m", limit=240)
                ob    = await self.exchange.fetch_order_book(s, limit=5)
                out[s] = {"ohlcv": ohlcv, "orderbook": ob}
            except Exception as e:
                # keep one-line warning, but don't spam
                from loguru import logger
                logger.warning(f"snapshot failed for {s}: {e}")
        return out
        
    async def trading_loop(self):
        self.loop_counter += 1
        if self.loop_counter % max(1, self.loop_log_every) == 0:
            logger.info(f"🔄 Trading Loop #{self.loop_counter} (Interval: {self.current_interval}s)")

        # Session
        if not self._in_session():
            logger.info(f"⏸ Outside session window (SESSION_HOURS_UTC={self.session_hours_utc}).")
            await asyncio.sleep(self.current_interval)
            return

        now = time.monotonic()
        due = []
        for name in self.cfg.strategies():
            interval = self.strategy_intervals.get(name, self.base_interval)
            nxt = self.strategy_next_run.get(name, 0.0)
            if now >= nxt:
                due.append(name)
        if not due:
            await asyncio.sleep(self.current_interval)
            return

        market = await self.fetch_market_snapshot()
        predictions = {}

        proposals: List[Proposal] = await self.strategy_manager.generate_proposals(
            market_data=market, predictions=predictions, only_strategies=due
        )

        for name in due:
            self.strategy_next_run[name] = now + max(1, self.strategy_intervals.get(name, self.base_interval))

        risk_budget = max(0.05 * (self.balance or self.initial_balance), 200.0)  # 5% or $200
        chosen = self.allocator.choose(proposals, risk_budget_usd=risk_budget, max_concurrent=2)
        logger.info(f"📊 {len(chosen)} chosen of {len(proposals)} proposals")

        for p in chosen:
            if not await self._pre_trade_guard(p):
                continue

            order = await self.exchange.place_order_with_sl_tp(
                symbol=p.symbol, side=p.side, quantity=(p.qty or 0.0),
                price=None, order_type="market",
                stop_loss=p.stop, take_profit=p.take, reduce_only=False,
                signal_strategy=p.strategy
            )
            if order:
                self.daily_trades += 1
                self.last_trade_ts[p.strategy] = time.time()
                # Track local position for trailing management
                self.positions[p.symbol] = {
                    "symbol": p.symbol,
                    "side": p.side.lower(),
                    "qty": p.qty,
                    "entry": p.entry,
                    "sl": p.stop,
                    "tp": p.take,
                    "current_sl": p.stop,
                    "peak": p.entry,
                    "trough": p.entry,
                    "breakeven_set": False,
                    "opened_ts": time.time(),
                }
                self.trade_history.append({
                    "ts": time.time(), "symbol": p.symbol, "side": p.side,
                    "strategy": p.strategy, "qty": p.qty, "entry": p.entry
                })
                logger.info(f"✅ Placed {p.strategy} {p.side.upper()} {p.symbol} qty={p.qty}")
            else:
                logger.warning(f"Order failed for {p.symbol}")

        await asyncio.sleep(self.current_interval)
