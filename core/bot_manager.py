# core/bot_manager.py
import os
import asyncio
from pathlib import Path
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from loguru import logger
from collections import Counter
from core.master_router import Proposal, EnsembleAllocator


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except Exception:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(float(os.getenv(key, default)))
    except Exception:
        return default


class BotManager:
    def _append_event(self, payload: dict) -> None:
        """Append one JSON line to logs/events.jsonl (best-effort)."""
        try:
            from pathlib import Path
            import json
            Path("logs").mkdir(exist_ok=True)
            with open("logs/events.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception as e:
            logger.debug(f"events.jsonl write failed: {e}")
            
    def __init__(self, config):
        self.cfg = config
        self.exchange = None

        # State
        self.active = False
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.balance = 0.0
        self.initial_balance = 0.0

        # Risk / performance knobs
        r = self.cfg.risk()
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.daily_loss_cap_pct = r["daily_loss_cap_pct"]
        self.max_daily_trades = r["max_daily_trades"]
        self.max_positions = r["max_positions"]
        self.min_rr = r["min_rr"]
        self.max_consec_losses = r["max_consec_losses"]
        self.loss_cooldown_min = r["loss_cooldown_min"]
        self.session_hours_utc = r["session_hours_utc"]

        # Cooldowns (per strategy)
        self.cooldowns = {
            "advanced_scalping": _env_int("COOLDOWN_ADV_SCALP", 60),
            "scalping": _env_int("COOLDOWN_SCALP", 60),
            "momentum": _env_int("COOLDOWN_MOMENTUM", 180),
            "mean_reversion": _env_int("COOLDOWN_MEANREV", 300),
            "pairs": _env_int("COOLDOWN_PAIRS", 120),
            "rsr": _env_int("COOLDOWN_RSR", 120),
        }
        self.last_trade_ts: Dict[str, float] = {}      # per-strategy last trade time
        self.last_trade_time: Dict[str, float] = {}    # per-symbol last trade time

        # Loop cadence
        self.base_interval = self.cfg.base_interval
        self.current_interval = self.base_interval
        self.loop_log_every = self.cfg.loop_log_every
        self.loop_counter = 0

        # Strategy cadence
        self.strategy_intervals = self.cfg.cadence()
        self.strategy_next_run = {k: 0.0 for k in self.strategy_intervals}

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
        self.max_spread_bps_scalp = _env_float("MAX_SPREAD_BPS_SCALP", 8.0)
        self.max_spread_bps_default = _env_float("MAX_SPREAD_BPS_DEFAULT", 12.0)
        self.min_depth_usd = _env_float("MIN_DEPTH_USD", 50_000.0)

        # For monitor: symbol -> strategy that opened it
        self.position_strategy = {}  # symbol -> strategy that opened it (for monitor)
        
    async def startup(self, exchange):
        self.exchange = exchange
        if self.cfg.force_one_way:
            await self.exchange.set_position_mode(one_way=True)

        # Balance
        try:
            bal = await self.exchange.exchange.fetch_balance()
            self.balance = float(
                bal.get("USDT", {}).get("total") or bal.get("USDT", {}).get("free") or 0.0
            )
        except Exception as e:
            logger.warning(f"fetch_balance on startup failed: {e}")
            self.balance = float(os.getenv("START_BALANCE", "10000"))
            self.initial_balance = self.balance or float(os.getenv("START_BALANCE", "10000"))

        logger.info("⚙️ Trade Control Settings:")
        logger.info(f"  Max positions total: {self.max_positions}")
        logger.info(f"  Max daily trades: {self.max_daily_trades}")
        logger.info(f"  Daily loss cap: -{self.daily_loss_cap_pct}%")
        logger.info(f"  Strategy intervals: {self.strategy_intervals}")
        logger.info(
            f"  Trailing: enabled={self.trailing_enabled}, "
            f"mode={self.trailing_cfg.mode}, "
            f"percent={self.trailing_cfg.percent}%, "
            f"breakeven_at={self.trailing_cfg.breakeven_at}%"
        )

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
        cur = self.balance or init
        pnl_usd = cur - init
        pnl_pct = (pnl_usd / init * 100.0) if init else 0.0

        # open positions (approx)
        open_pos = 0
        try:
            poss = await self.exchange.exchange.fetch_positions(params={"category": "linear"})
            for p in poss or []:
                info = p.get("info") or {}
                size = 0.0
                for k in ("contracts", "contractSize", "positionAmt"):
                    v = p.get(k)
                    if v is not None:
                        try:
                            size = float(v)
                            break
                        except Exception:
                            pass
                if not size:
                    for k in ("size", "positionValue", "positionAmt"):
                        v = info.get(k)
                        if v is not None:
                            try:
                                size = float(v)
                                break
                            except Exception:
                                pass
                if abs(size) > 0:
                    open_pos += 1
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
        """
        Maintain trailing stops & breakeven for open positions.
        Also emits a single 'exit' event to logs/events.jsonl when an exchange position disappears.

        Exit PnL:
          1) If ExchangeClient exposes a realized PnL accessor, we try that first.
          2) Otherwise we approximate using current mark vs entry * qty with correct sign.
        """
        for sym in list(self.positions.keys()):  # copy keys to avoid dict mutation issues
            pos = self.positions.get(sym)
            if not pos:
                continue

            # 1) Exit handling: position disappeared on exchange
            still_open = await self.exchange.has_open_position(sym)
            if not still_open:
                logger.info(f"Position closed on exchange: {sym}. Removing from local state.")

                stg = self.position_strategy.pop(sym, None)
                pos_info = pos or {}

                # Try to get realized PnL from the exchange if available
                realized = None
                try:
                    fetch_realized = getattr(self.exchange, "fetch_realized_pnl", None)
                    if callable(fetch_realized):
                        opened_ts = int((pos_info.get("opened_ts") or time.time()) * 1000)
                        realized = await fetch_realized(sym, since_ms=opened_ts)
                except Exception:
                    realized = None

                # Fallback: rough estimate from last price vs entry
                if realized is None:
                    try:
                        entry = float(
                            pos_info.get("entry")
                            or pos_info.get("entry_price")
                            or pos_info.get("avg_entry_price")
                            or 0.0
                        )
                    except Exception:
                        entry = 0.0
                    try:
                        qty = float(
                            pos_info.get("qty")
                            or pos_info.get("contracts")
                            or pos_info.get("size")
                            or 0.0
                        )
                    except Exception:
                        qty = 0.0
                    side = (pos_info.get("side") or "").lower()

                    try:
                        last = await self.exchange.get_price(sym)
                        if entry and qty and last:
                            if side in ("buy", "long"):
                                realized = (last - entry) * qty
                            elif side in ("sell", "short"):
                                realized = (entry - last) * qty
                    except Exception:
                        realized = None

                # Emit a single 'exit' event for the monitor
                self._append_event({
                    "ts": time.time(),
                    "type": "exit",
                    "strategy": stg,
                    "symbol": sym,
                    "realized_pnl": float(realized) if realized is not None else None,
                })

                # Remove local state
                self.positions.pop(sym, None)
                continue

            # 2) Still open: manage trailing logic
            price = await self.exchange.get_price(sym)
            if price is None:
                continue

            # Safe coercions
            try:
                side = (pos.get("side") or "").lower()
                entry = float(pos.get("entry") or pos.get("entry_price") or 0.0)
                cur_sl = pos.get("current_sl")
                cur_sl = float(cur_sl) if cur_sl is not None else None
                tp = pos.get("tp")
                tp = float(tp) if tp is not None else None
                trail_pct = float(self.trailing_cfg.percent)
                breakeven_at = float(self.trailing_cfg.breakeven_at)
                price = float(price)
            except Exception:
                continue

            if side in ("buy", "long"):
                # Track peak
                base = entry or price
                pos["peak"] = max(float(pos.get("peak", base)), price)
                up_pct = ((pos["peak"] - entry) / entry * 100.0) if entry else 0.0

                # Breakeven once in profit by X%
                if not pos.get("breakeven_set", False) and up_pct >= breakeven_at and entry:
                    new_sl = max(float(pos.get("sl") or 0.0), entry)
                    if cur_sl is None or new_sl > cur_sl:
                        ok = await self.exchange.set_stop_loss(sym, new_sl)
                        if ok:
                            pos["current_sl"] = new_sl
                            pos["breakeven_set"] = True
                            logger.info(f"Breakeven set for {sym} @ {new_sl:.6f}")

                # Trail from peak by trail_pct; never trail above TP
                trail_stop = pos["peak"] * (1 - trail_pct / 100.0)
                if tp is not None:
                    trail_stop = min(trail_stop, tp)

                if (cur_sl is None or trail_stop > cur_sl) and trail_stop < price:
                    ok = await self.exchange.set_stop_loss(sym, trail_stop)
                    if ok:
                        pos["current_sl"] = trail_stop
                        logger.info(f"Trailing SL moved up for {sym} -> {trail_stop:.6f}")

            elif side in ("sell", "short"):
                # Track trough
                base = entry or price
                pos["trough"] = min(float(pos.get("trough", base)), price)
                down_pct = ((entry - pos["trough"]) / entry * 100.0) if entry else 0.0

                # Breakeven for shorts
                if not pos.get("breakeven_set", False) and down_pct >= breakeven_at and entry:
                    new_sl = min(float(pos.get("sl") or 1e18), entry)
                    if cur_sl is None or new_sl < cur_sl:
                        ok = await self.exchange.set_stop_loss(sym, new_sl)
                        if ok:
                            pos["current_sl"] = new_sl
                            pos["breakeven_set"] = True
                            logger.info(f"Breakeven set for {sym} (SHORT) @ {new_sl:.6f}")

                # Trail from trough by trail_pct; never trail below TP (short)
                trail_stop = pos["trough"] * (1 + trail_pct / 100.0)
                if tp is not None:
                    trail_stop = max(trail_stop, tp)

                if (cur_sl is None or trail_stop < cur_sl) and trail_stop > price:
                    ok = await self.exchange.set_stop_loss(sym, trail_stop)
                    if ok:
                        pos["current_sl"] = trail_stop
                        logger.info(f"Trailing SL moved down for {sym} (SHORT) -> {trail_stop:.6f}")

  
    # ---------- Helpers ----------
    def _append_event(self, payload: dict) -> None:
        """Append a compact JSON event line for tools/monitor.py."""
        try:
            from pathlib import Path
            import json, time as _t
            Path("logs").mkdir(exist_ok=True)
            payload = dict(payload)
            payload.setdefault("ts", _t.time())
            with open("logs/events.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception:
            pass
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
        if not entry or not sl or not tp:
            return 0.0
        if side.lower() == "buy":
            risk = max(entry - sl, 1e-9)
            reward = max(tp - entry, 0.0)
        else:
            risk = max(sl - entry, 1e-9)
            reward = max(entry - tp, 0.0)
        return reward / risk if risk > 0 else 0.0
    def _log_entry_event(self, proposal, qty: float, price, order: dict) -> None:
        """Write an 'entry' event the monitor can read."""
        try:
            Path("logs").mkdir(exist_ok=True)
            payload = {
                "ts": time.time(),
                "type": "entry",
                "strategy": getattr(proposal, "strategy", None),
                "symbol": getattr(proposal, "symbol", None),
                "side": str(getattr(proposal, "side", "")).upper(),
                "qty": float(qty),
                "entry": (float(price) if price is not None else None),
                "order_id": (order.get("id") if isinstance(order, dict) else None),
            }
            with open("logs/events.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception as e:
            logger.debug(f"Failed to write entry event: {e}")
    # def _append_event(self, payload: dict) -> None:
        # try:
            # Path("logs").mkdir(exist_ok=True)
            # with open("logs/events.jsonl", "a", encoding="utf-8") as f:
                # f.write(json.dumps(payload) + "\n")
        # except Exception as e:
            # logger.warning(f"[events] write failed: {e}")
   
            
    async def _pre_trade_guard(self, p: Proposal) -> bool:
        # daily loss cap
        if self.initial_balance > 0 and self.balance > 0:
            dd_pct = (self.initial_balance - self.balance) / self.initial_balance * 100.0
            if dd_pct >= self.daily_loss_cap_pct:
                logger.error(
                    f"⛔ Daily loss cap hit ({dd_pct:.2f}% ≥ {self.daily_loss_cap_pct}%). Pausing entries."
                )
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
        max_spread = (
            self.max_spread_bps_scalp
            if p.strategy in ("scalping", "advanced_scalping")
            else self.max_spread_bps_default
        )
        depth_ok = (bid_depth >= self.min_depth_usd and ask_depth >= self.min_depth_usd)
        if spread_bps > max_spread or not depth_ok:
            logger.info(
                f"⛔ Microstructure guard: spread={spread_bps:.1f}bps (max {max_spread}), "
                f"bid_depth=${bid_depth:,.0f}, ask_depth=${ask_depth:,.0f} (min ${self.min_depth_usd:,.0f})"
            )
            return False

        return True

    # ---------- Market snapshot ----------
    async def fetch_market_snapshot(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"_bar_index": int(time.time() // 60)}
        markets = (self.exchange.exchange.markets or {})
        for s in self.cfg.symbols_list():
            if s not in markets:
                # silently skip unknown/unsupported symbols on this venue/testnet
                continue
            try:
                ohlcv = await self.exchange.exchange.fetch_ohlcv(s, timeframe="1m", limit=240)
                ob = await self.exchange.fetch_order_book(s, limit=5)
                out[s] = {"ohlcv": ohlcv, "orderbook": ob}
            except Exception as e:
                logger.warning(f"snapshot failed for {s}: {e}")
        return out

    def _append_event(self, payload: dict) -> None:
            """
            Append one JSON line to logs/events.jsonl for the monitor.
            Safe: creates folder if missing and ignores write errors.
            """
            try:
                Path("logs").mkdir(exist_ok=True)
                with open("logs/events.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload) + "\n")
            except Exception:
                # don't crash trading loop if disk/permission issue
                pass


    async def _safe_set_sl(self, symbol: str, desired: float, side: str) -> bool:
        """
        Clamp & round SL so Bybit accepts it, then call exchange.set_stop_loss().
        - For LONG: SL must be < last price (with a small safety gap)
        - For SHORT: SL must be > last price (with a small safety gap)
        - Rounds to market tick size when available (ccxt market meta)
        """
        try:
            last = await self.exchange.get_price(symbol)
            if last is None:
                return False

            # small safety buffer so SL is not equal/invalid vs base/last
            gap_bps = float(os.getenv("SL_SAFETY_GAP_BPS", "5"))  # 5 bps = 0.05%

            if str(side).lower() in ("buy", "long"):
                # must be below last
                max_sl = last * (1.0 - gap_bps / 10000.0)
                target = min(float(desired), max_sl)
                round_up = False
            else:
                # must be above last
                min_sl = last * (1.0 + gap_bps / 10000.0)
                target = max(float(desired), min_sl)
                round_up = True

            # round to tick size if we can get it from the ccxt market info
            tick = 0.0
            try:
                m = self.exchange.exchange.market(symbol)  # ccxt market meta
                tick = float(
                    (m.get("precision", {}) or {}).get("price")
                    or (m.get("limits", {}).get("price", {}) or {}).get("step")
                    or (m.get("limits", {}).get("price", {}) or {}).get("min")
                    or 0.0
                )
            except Exception:
                tick = 0.0

            if tick and tick > 0:
                steps = math.ceil(target / tick) if round_up else math.floor(target / tick)
                target = steps * tick

            # avoid spamming tiny moves
            cur_sl = (self.positions.get(symbol) or {}).get("current_sl")
            if cur_sl is not None and tick and abs(float(cur_sl) - float(target)) < tick:
                return True

            ok = await self.exchange.set_stop_loss(symbol, float(target))
            if ok and symbol in self.positions:
                self.positions[symbol]["current_sl"] = float(target)
            return bool(ok)

        except Exception as e:
            logger.error(f"_safe_set_sl error for {symbol}: {e}")
            return False
    async def _manage_all_trails(self):
        """
        - If a position disappears on exchange → emit 'exit' event (with best-effort realized PnL) and drop local state.
        - Maintain breakeven + trailing SL, using exchange-side clamp to avoid Bybit 10001 errors.
        """
        for sym in list(self.positions.keys()):
            pos = self.positions.get(sym)
            if not pos:
                continue

            # 1) Gone on exchange? => write exit event + drop
            try:
                still_open = await self.exchange.has_open_position(sym)
            except Exception:
                continue

            if not still_open:
                stg = self.position_strategy.pop(sym, None)
                pos_info = pos or {}

                # best-effort realized pnl using last price (if exchange can't give us realized)
                realized = None
                try:
                    last  = await self.exchange.get_price(sym)
                    entry = float(pos_info.get("entry") or 0.0)
                    qty   = float(pos_info.get("qty")   or 0.0)
                    side  = (pos_info.get("side") or "").lower()
                    if entry and qty and last:
                        realized = (last - entry) * qty if side in ("buy","long") else (entry - last) * qty
                    else:
                        realized = 0.0
                except Exception:
                    realized = 0.0

                self._append_event({
                    "ts": time.time(),
                    "type": "exit",
                    "strategy": stg,
                    "symbol": sym,
                    "realized_pnl": float(realized),
                })

                self.positions.pop(sym, None)
                continue

            # 2) Still open: move SLs safely
            price = await self.exchange.get_price(sym)
            if price is None:
                continue

            side = str(pos.get("side", "")).lower()
            entry = float(pos.get("entry") or 0.0)
            cur_sl = pos.get("current_sl")
            tp = pos.get("tp")

            trail_pct = float(self.trailing_cfg.percent)
            breakeven_at = float(self.trailing_cfg.breakeven_at)

            if side in ("buy", "long"):
                pos["peak"] = max(float(pos.get("peak", entry or price)), float(price))
                up_pct = ((pos["peak"] - entry) / entry * 100.0) if entry else 0.0

                # breakeven
                if not pos.get("breakeven_set", False) and up_pct >= breakeven_at and entry:
                    desired = max(float(pos.get("sl") or 0.0), float(entry))
                    ok = await self.exchange.set_stop_loss(sym, desired, side="long")
                    if ok:
                        pos["current_sl"] = desired
                        pos["breakeven_set"] = True

                # trail from peak
                trail_stop = float(pos["peak"]) * (1 - trail_pct / 100.0)
                if tp is not None:
                    try:
                        trail_stop = min(trail_stop, float(tp))
                    except Exception:
                        pass

                if (cur_sl is None or trail_stop > float(cur_sl)) and trail_stop < float(price):
                    ok = await self.exchange.set_stop_loss(sym, trail_stop, side="long")
                    if ok:
                        pos["current_sl"] = trail_stop

            elif side in ("sell", "short"):
                base = entry or price or 0.0
                pos["trough"] = min(float(pos.get("trough", base)), float(price))
                down_pct = ((entry - pos["trough"]) / entry * 100.0) if entry else 0.0

                # breakeven for shorts
                if not pos.get("breakeven_set", False) and down_pct >= breakeven_at and entry:
                    desired = min(float(pos.get("sl") or 1e18), float(entry))
                    ok = await self.exchange.set_stop_loss(sym, desired, side="short")
                    if ok:
                        pos["current_sl"] = desired
                        pos["breakeven_set"] = True

                # trail from trough
                trail_stop = float(pos["trough"]) * (1 + trail_pct / 100.0)
                if tp is not None:
                    try:
                        trail_stop = max(trail_stop, float(tp))
                    except Exception:
                        pass

                if (cur_sl is None or trail_stop < float(cur_sl)) and trail_stop > float(price):
                    ok = await self.exchange.set_stop_loss(sym, trail_stop, side="short")
                    if ok:
                        pos["current_sl"] = trail_stop


    # ---------- Main loop ----------
    async def trading_loop(self):
        self.loop_counter += 1
        if self.loop_counter % max(1, self.loop_log_every) == 0:
            logger.info(f"🔄 Trading Loop #{self.loop_counter} (Interval: {self.current_interval}s)")

        # Session window guard
        if not self._in_session():
            logger.info(f"⏸ Outside session window (SESSION_HOURS_UTC={self.session_hours_utc}).")
            await asyncio.sleep(self.current_interval)
            return

        # Which strategies are due this tick?
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

        # Market + proposals
        market = await self.fetch_market_snapshot()
        predictions = {}
        proposals: List[Proposal] = await self.strategy_manager.generate_proposals(
            market_data=market, predictions=predictions, only_strategies=due
        )

        # Quick summary of proposals by strategy
        by_strat = Counter(getattr(p, "strategy", "?") for p in proposals)
        if by_strat:
            summary = ", ".join(f"{k}={v}" for k, v in by_strat.items())
            logger.info(f"🗂️ Proposals by strategy: {summary}")
        else:
            logger.info("🗂️ Proposals by strategy: (none)")

        # Next run scheduling for due strategies
        for name in due:
            self.strategy_next_run[name] = now + max(1, self.strategy_intervals.get(name, self.base_interval))

        # Choose which to execute
        risk_budget = max(0.05 * (self.balance or self.initial_balance or 0.0), 200.0)  # 5% or $200
        chosen = self.allocator.choose(proposals, risk_budget_usd=risk_budget, max_concurrent=2)
        logger.info(f"📊 {len(chosen)} chosen of {len(proposals)} proposals")

        chosen_by = Counter(getattr(p, "strategy", "?") for p in chosen)
        if chosen_by:
            chosen_summary = ", ".join(f"{k}={v}" for k, v in chosen_by.items())
            logger.info(f"🎯 Chosen by strategy: {chosen_summary}")
        else:
            logger.info("🎯 Chosen by strategy: (none)")

        # Execute chosen proposals
        for p in chosen:
            # pre-trade guards (RR floor, cooldowns, microstructure, caps…)
            if not await self._pre_trade_guard(p):
                continue

            # Place order
            try:
                order = await self.exchange.place_order_with_sl_tp(
                    symbol=p.symbol,
                    side=p.side,
                    quantity=float(p.qty or 0.0),
                    price=None,                # market
                    order_type="market",
                    stop_loss=p.stop,
                    take_profit=p.take,
                    reduce_only=False,
                )
            except Exception as e:
                logger.error(f"Order placement failed for {p.symbol}: {e}")
                order = None

            if not order:
                logger.error(f"Order placement returned None for {p.symbol}")
                continue

            # Record success
            ts_now = time.time()
            self.daily_trades = (self.daily_trades or 0) + 1
            self.last_trade_ts[p.strategy] = ts_now
            self.last_trade_time[p.symbol] = ts_now

            # Safely pull fields from 'order' (ccxt dict)
            if isinstance(order, dict):
                order_id = order.get("id")
                entry_filled = order.get("average") or order.get("price") or p.entry
                amount_filled = order.get("amount")
            else:
                order_id = None
                entry_filled = p.entry
                amount_filled = None

            try:
                entry_filled = float(entry_filled) if entry_filled is not None else 0.0
            except Exception:
                entry_filled = 0.0

            qty_val = float(p.qty or amount_filled or 0.0)

            # Update local position cache
            self.positions[p.symbol] = {
                "symbol": p.symbol,
                "side": str(p.side).lower(),
                "qty": qty_val,
                "entry": entry_filled,
                "sl": float(p.stop) if p.stop is not None else None,
                "tp": float(p.take) if p.take is not None else None,
                "current_sl": float(p.stop) if p.stop is not None else None,
                "peak": entry_filled,
                "trough": entry_filled,
                "breakeven_set": False,
                "opened_ts": ts_now,
                "order_id": order_id,
            }

            # History line (optional)
            self.trade_history.append(
                {
                    "ts": ts_now,
                    "order_id": order_id,
                    "symbol": p.symbol,
                    "side": str(p.side).upper(),
                    "strategy": p.strategy,
                    "qty": qty_val,
                    "entry": entry_filled,
                    "sl": float(p.stop) if p.stop is not None else None,
                    "tp": float(p.take) if p.take is not None else None,
                }
            )

            logger.info(f"✅ Placed {p.strategy} {str(p.side).upper()} {p.symbol} qty={qty_val} (order {order_id or 'n/a'})")

            # Remember which strategy opened this symbol (for the monitor)
            self.position_strategy[p.symbol] = p.strategy

            # Write monitor entry event
        try:
            Path("logs").mkdir(exist_ok=True)
            with open("logs/events.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": ts_now,
                    "type": "entry",
                    "strategy": p.strategy,
                    "symbol": p.symbol,
                    "side": str(p.side).upper(),
                    "qty": qty_val,
                    "entry": entry_filled,
                    "order_id": order_id
                }) + "\n")
        except Exception as e:
            logger.debug(f"Failed to write entry event: {e}")

            # keep trailing stops fresh + emit 'exit' events when a position disappears on the exchange
                
        try:
            await self._manage_all_trails()
        except Exception as e:
            logger.debug(f"_manage_all_trails error: {e}")
                                         
     # sleep between loops
        await asyncio.sleep(self.current_interval)

      
