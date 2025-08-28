import os, time, asyncio, json
from typing import Any, Dict, Optional, List, Tuple
from loguru import logger
#from typing import Optional
import ccxt.async_support as ccxt  # async ccxt



class ExchangeClient:
    """
    Async wrapper around ccxt.bybit (v5).
    - Sandbox toggle, One-Way position mode
    - Price & orderbook, spread/depth metrics
    - Robust order placement with tpslMode, price-band clamp, positionIdx retry
    - Post-only maker preference for non-scalp
    - Helpers: has_open_position, reduce_position
    """
    def __init__(self, config):
        self.config = config
        self.exchange: Optional[ccxt.Exchange] = None
        self.orders: Dict[str, Dict[str, Any]] = {}

    async def connect(self):
        self.exchange = ccxt.bybit({
            "apiKey": os.getenv("BYBIT_API_KEY"),
            "secret": os.getenv("BYBIT_API_SECRET"),
            "options": {
                "defaultType": "swap",
                "recvWindow": 60000,
            }
        })
        self.exchange.set_sandbox_mode(self.config.testnet)
        logger.info(f"Connecting to Bybit {'Testnet' if self.config.testnet else 'Live'}...")
        await self.exchange.load_markets()
        logger.info(f"Markets loaded: {len(self.exchange.markets)}")

        try:
            bal = await self.exchange.fetch_balance()
            usdt = bal.get("USDT", {}).get("total") or bal.get("USDT", {}).get("free")
            logger.info(f"Account balance: {usdt} USDT")
        except Exception as e:
            logger.warning(f"Could not fetch balance: {e}")

    async def close(self):
        if self.exchange:
            await self.exchange.close()

    async def set_position_mode(self, one_way: bool = True):
        try:
            await self.exchange.set_position_mode(not one_way, params={"category": "linear"})
            logger.info(f"Set Bybit position mode to {'One-Way' if one_way else 'Hedge'} for linear perps")
        except Exception as e:
            logger.warning(f"Could not set position mode globally: {e}")

    async def get_price(self, symbol: str) -> Optional[float]:
        try:
            t = await self.exchange.fetch_ticker(symbol)
            return float(t["last"]) if t and t.get("last") else None
        except Exception as e:
            logger.error(f"get_price error for {symbol}: {e}")
            return None

# --- helpers: tick size / quantization / clamping ---

    def _get_market(self, symbol: str):
        try:
            return self.exchange.market(symbol)
        except Exception:
            # fallback: assume markets were loaded
            return self.exchange.markets.get(symbol, {})

    def get_tick_size(self, symbol: str) -> float:
        m = self._get_market(symbol)
        # bybit usually: m["info"]["priceFilter"]["tickSize"]
        try:
            ts = float(m["info"]["priceFilter"]["tickSize"])
            if ts > 0:
                return ts
        except Exception:
            pass
        # fallback from precision
        try:
            prec = m.get("precision", {}).get("price")
            if prec is not None:
                return 10 ** (-int(prec))
        except Exception:
            pass
        # ultra fallback
        return 0.01

    def quantize_price(self, symbol: str, price: float) -> float:
        step = self.get_tick_size(symbol)
        if step <= 0:
            return float(price)
        # floor to the nearest tick
        return float((int(price / step)) * step)

    def clamp_stop_for_side(self, symbol: str, side: str, desired: float, last: float) -> float | None:
        """
        For longs: SL must be < last
        For shorts: SL must be > last
        Returns a safe (quantized) SL or None if no valid SL can be made.
        """
        side = (side or "").lower()
        step = self.get_tick_size(symbol)
        epsilon = step * 2.0 if step > 0 else 0.01

        if last is None or desired is None:
            return None

        if side in ("buy", "long"):
            safe = min(float(desired), float(last) - epsilon)
            if not (safe < last):
                return None
        else:  # "sell", "short"
            safe = max(float(desired), float(last) + epsilon)
            if not (safe > last):
                return None

        return self.quantize_price(symbol, safe)


    async def fetch_order_book(self, symbol: str, limit: int = 5):
        try:
            return await self.exchange.fetch_order_book(symbol, limit=limit)
        except Exception as e:
            logger.error(f"fetch_order_book error {symbol}: {e}")
            return {"bids": [], "asks": []}

    async def get_spread_metrics(self, symbol: str) -> Tuple[float, float, float]:
        ob = await self.fetch_order_book(symbol, limit=3)
        if not ob.get("bids") or not ob.get("asks"):
            return (1e9, 0.0, 0.0)
        best_bid = float(ob["bids"][0][0]); best_ask = float(ob["asks"][0][0])
        mid = (best_bid + best_ask) / 2.0
        spread_bps = (best_ask - best_bid) / mid * 1e4 if mid else 1e9
        bid_depth_usd = sum(float(p)*float(q) for p, q in ob["bids"])
        ask_depth_usd = sum(float(p)*float(q) for p, q in ob["asks"])
        return (spread_bps, bid_depth_usd, ask_depth_usd)

    # Unified wrappers
    async def create_market_order(self, symbol, side, amount, params=None):
        return await self.exchange.create_order(symbol, "market", side, amount, None, params or {})

    async def create_limit_order(self, symbol, side, amount, price, params=None):
        return await self.exchange.create_order(symbol, "limit", side, amount, price, params or {})
    
        # ---- Normalize quantity to exchange limits/precision ----
        try:
            market = self.exchange.market(symbol)
        except Exception:
            market = None

        try:
            # Clamp to min amount if available
            min_amt = None
            if market:
                min_amt = ((market.get("limits") or {}).get("amount") or {}).get("min")
            if min_amt is not None:
                min_amt = float(min_amt)
                if quantity < min_amt:
                    logger.warning(f"Clamping qty from {quantity} -> min {min_amt} for {symbol}")
                    quantity = min_amt

            # Round to exchange precision (ccxt helper)
            quantity = float(self.exchange.amount_to_precision(symbol, quantity))
        except Exception as _e:
            logger.debug(f"qty normalize skipped for {symbol}: {_e}")
        # ---- end normalize ----


    async def set_stop_loss(self, symbol: str, price: float, side: str | None = None) -> bool:
        """
        Set SL using Bybit's trading stop.
        - side is required to clamp correctly (long vs short).
        """
        try:
            last = await self.get_price(symbol)
            if last is None:
                self.logger.error(f"set_stop_loss: no last price for {symbol}")
                return False

            safe = self.clamp_stop_for_side(symbol, side or "", price, last)
            if safe is None:
                self.logger.warning(
                    f"set_stop_loss: requested SL {price} invalid against last={last} for {symbol} ({side}), skipping."
                )
                return False

            # quantized safe already
            params = {}
            # your existing call here (unified ccxt)
            # Prefer: setTradingStop or editPosition 'stopLoss' depending on your code
            # Example:
            resp = await self.exchange.setTradingStop(symbol, stopLoss=safe, params=params)
            self.logger.info(f"set_stop_loss: {symbol} -> {safe} (side={side})")
            return True
        except Exception as e:
            self.logger.error(f"set_stop_loss failed: {e}")
            return False

    async def set_take_profit(self, symbol: str, price: float) -> bool:
        try:
            market = self.exchange.market(symbol)
            req = {
                "category": "linear",
                "symbol": market["id"],
                "takeProfit": str(round(float(price), 6)),
                "tpTriggerBy": "LastPrice",
            }
            if hasattr(self.exchange, "privatePostV5PositionTradingStop"):
                await self.exchange.privatePostV5PositionTradingStop(req)
            elif hasattr(self.exchange, "privatePostContractV3PrivatePositionTradingStop"):
                await self.exchange.privatePostContractV3PrivatePositionTradingStop(req)
            else:
                logger.warning("No trading_stop endpoint found in ccxt; skipping set_take_profit")
                return False
            return True
        except Exception as e:
            logger.error(f"set_take_profit failed: {e}")
            return False

    async def has_open_position(self, symbol: str) -> bool:
        """Check if there's a non-zero position size for this symbol."""
        try:
            try:
                poss = await self.exchange.fetch_positions(symbols=[symbol], params={"category":"linear"})
            except Exception:
                poss = await self.exchange.fetch_positions(params={"category":"linear"})
            for p in poss or []:
                sym = p.get("symbol") or p.get("info", {}).get("symbol", "")
                if sym and symbol.split(":")[0] not in sym:
                    continue
                size = 0.0
                for k in ("contracts","contractSize","positionAmt"):
                    v = p.get(k)
                    if v is not None:
                        try: size = float(v); break
                        except Exception: pass
                if not size:
                    info = p.get("info") or {}
                    for k in ("size","positionValue","positionAmt"):
                        v = info.get(k)
                        if v is not None:
                            try: size = float(v); break
                            except Exception: pass
                if abs(size) > 0:
                    return True
        except Exception:
            pass
        return False

    async def reduce_position(self, symbol: str, side: str, amount: float) -> bool:
        """
        Close amount using reduceOnly market order.
        Pass side='sell' to close a long, side='buy' to close a short.
        """
        try:
            params = {"reduceOnly": True, "category": "linear"}
            o = await self.create_market_order(symbol, side, amount, params=params)
            if o:
                logger.info(f"✅ Reduced position {symbol} by {amount} via {side.upper()} reduce-only")
                return True
        except Exception as e:
            logger.error(f"reduce_position failed {symbol}: {e}")
        return False

    # --- robust place_order_with_sl_tp unchanged except minor context ---
    async def place_order_with_sl_tp(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "market",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        reduce_only: bool = False,
        signal_strategy: str = "unknown",
    ) -> Optional[Dict]:
        """
        Robust Bybit order with SL/TP attachment and resilient retries (price band, positionIdx, strip+set).
        """
        try:
            current_price = await self.get_price(symbol)
            if not current_price or current_price <= 0:
                logger.error(f"Invalid current price for {symbol}: {current_price}")
                return None

            logger.info(f"Current market price for {symbol}: ${current_price:,.2f}")

            side_l = side.lower()
            order_type_l = (order_type or "market").lower()
            if order_type_l not in ("market", "limit"):
                order_type_l = "market"

            # Direction sanity + gentle auto-corrects
            if side_l in ("buy", "long"):
                if stop_loss is not None and stop_loss >= current_price:
                    stop_loss = current_price * 0.98
                if take_profit is not None and take_profit <= current_price:
                    take_profit = current_price * 1.03
                if order_type_l == "limit" and price and price > current_price:
                    price = current_price * 0.999
            else:
                if stop_loss is not None and stop_loss <= current_price:
                    stop_loss = current_price * 1.02
                if take_profit is not None and take_profit >= current_price:
                    take_profit = current_price * 0.97
                if order_type_l == "limit" and price and price < current_price:
                    price = current_price * 1.001

            params: Dict[str, Any] = {"category": "linear"}
            if reduce_only:
                params["reduceOnly"] = True

            has_sl = stop_loss is not None
            has_tp = take_profit is not None
            if has_sl or has_tp:
                params["tpslMode"] = "Full"
            if has_sl:
                params["stopLoss"]   = str(round(float(stop_loss), 6))
                params["slTriggerBy"] = "LastPrice"
                params["slOrderType"] = "Market"
            if has_tp:
                params["takeProfit"] = str(round(float(take_profit), 6))
                params["tpTriggerBy"] = "LastPrice"
                params["tpOrderType"] = "Market"

            logger.info(
                f"Placing {order_type_l.upper()} {side_l.upper()} order:\n"
                f"  Symbol: {symbol}\n"
                f"  Qty: {quantity}\n"
                f"  Price: {('MKT' if price is None else f'${price:.6f}')}\n"
                f"  SL: {('—' if stop_loss is None else f'${stop_loss:.6f}')}\n"
                f"  TP: {('—' if take_profit is None else f'${take_profit:.6f}')}"
            )

            async def _submit(p: Dict[str, Any], order_price: Optional[float] = None) -> Optional[Dict]:
                if order_type_l == "market":
                    return await self.create_market_order(symbol, side_l, quantity, params=p)
                else:
                    use_price = order_price if order_price is not None else (price if price is not None else current_price)
                    p2 = dict(p)
                    if signal_strategy not in ("scalping", "advanced_scalping"):
                        p2["postOnly"] = True
                        try:
                            return await self.create_limit_order(symbol, side_l, quantity, use_price, params=p2)
                        except Exception:
                            p2.pop("postOnly", None)
                    return await self.create_limit_order(symbol, side_l, quantity, use_price, params=p2)

            safe_limit_price: Optional[float] = None

            try:
                order = await _submit(params)
                if order:
                    self.orders[order.get("id", f"{symbol}-{time.time()}")] = order
                    logger.info(f"✅ Order placed successfully: {order.get('id','<no-id>')}")
                    return order
            except Exception as first_err:
                emsg = str(first_err)
                lower = emsg.lower()
                ret_code = None
                try:
                    jtxt = emsg.split("bybit", 1)[1].strip()
                    jobj = json.loads(jtxt)
                    ret_code = int(jobj.get("retCode"))
                except Exception:
                    pass

                if "position idx not match position mode" in lower:
                    logger.warning("Bybit: positionIdx mismatch. Retrying in Hedge mode with positionIdx...")
                    hedge_params = dict(params)
                    hedge_params["positionIdx"] = 1 if side_l in ("buy","long") else 2
                    try:
                        order = await _submit(hedge_params)
                        if order:
                            self.orders[order.get("id", f"{symbol}-{time.time()}")] = order
                            logger.info(f"✅ Order placed on retry (Hedge positionIdx): {order.get('id','<no-id>')}")
                            return order
                    except Exception as second_err:
                        logger.error(f"Retry with positionIdx failed: {second_err}")

                if ret_code == 30208 or "maximum buying price" in lower or "minimum selling price" in lower:
                    logger.warning(f"Price band hit: {emsg}. Fetching top-of-book to clamp price and retry as LIMIT...")
                    try:
                        ob = await self.fetch_order_book(symbol, limit=1)
                        best_bid = ob["bids"][0][0] if ob.get("bids") else current_price
                        best_ask = ob["asks"][0][0] if ob.get("asks") else current_price

                        if side_l in ("buy","long"):
                            safe_limit_price = best_ask * 1.0002
                            if has_tp and take_profit is not None and take_profit <= safe_limit_price:
                                take_profit = safe_limit_price * 1.002
                            if has_sl and stop_loss is not None and stop_loss >= safe_limit_price:
                                stop_loss = safe_limit_price * 0.995
                        else:
                            safe_limit_price = best_bid * 0.9998
                            if has_tp and take_profit is not None and take_profit >= safe_limit_price:
                                take_profit = safe_limit_price * 0.998
                            if has_sl and stop_loss is not None and stop_loss <= safe_limit_price:
                                stop_loss = safe_limit_price * 1.005

                        try:
                            safe_limit_price = float(self.exchange.price_to_precision(symbol, safe_limit_price))
                        except Exception:
                            pass

                        params_clamped = dict(params)
                        if has_tp and take_profit is not None:
                            try: take_profit = float(self.exchange.price_to_precision(symbol, take_profit))
                            except Exception: pass
                            params_clamped["takeProfit"] = str(round(float(take_profit), 6))
                        if has_sl and stop_loss is not None:
                            try: stop_loss = float(self.exchange.price_to_precision(symbol, stop_loss))
                            except Exception: pass
                            params_clamped["stopLoss"] = str(round(float(stop_loss), 6))

                        saved_type = order_type_l
                        order_type_l = "limit"
                        try:
                            order = await _submit(params_clamped, order_price=safe_limit_price)
                            if order:
                                self.orders[order.get("id", f"{symbol}-{time.time()}")] = order
                                logger.info(f"✅ Order placed after price clamp: {order.get('id','<no-id>')}")
                                return order
                        finally:
                            order_type_l = saved_type
                    except Exception as clamp_err:
                        logger.error(f"Retry with clamped price failed: {clamp_err}")

                must_strip_tpsl = any(x in lower for x in (
                    "tpslmode","tpordertype","slordertype",
                    "tpordertype can not have a value when","param error"
                )) or "position idx not match position mode" in lower or ret_code == 30208

                if must_strip_tpsl:
                    logger.warning(f"Retrying order WITHOUT in-order TP/SL, will set via trading_stop... ({emsg})")

                    def _strip_tpsl(p: Dict[str, Any]) -> Dict[str, Any]:
                        keys = ("tpslMode","stopLoss","slTriggerBy","slOrderType","takeProfit","tpTriggerBy","tpOrderType")
                        return {k: v for k, v in p.items() if k not in keys}

                    v1 = _strip_tpsl(params)
                    v2 = _strip_tpsl(params); v2["positionIdx"] = 1 if side_l in ("buy","long") else 2

                    for i, ptry in enumerate((v1, v2), 1):
                        try:
                            if safe_limit_price is not None:
                                saved_type = order_type_l
                                order_type_l = "limit"
                                try:
                                    order = await _submit(ptry, order_price=safe_limit_price)
                                finally:
                                    order_type_l = saved_type
                            else:
                                order = await _submit(ptry)

                            if order:
                                self.orders[order.get("id", f"{symbol}-{time.time()}")] = order
                                logger.info(f"✅ Order placed on retry {i} (no in-order TP/SL): {order.get('id','<no-id>')}")
                                await asyncio.sleep(1.0)
                                if has_sl and stop_loss is not None:
                                    ok_sl = await self.set_stop_loss(symbol, float(stop_loss))
                                    if not ok_sl: logger.warning("Could not set stop loss via trading_stop")
                                if has_tp and take_profit is not None:
                                    ok_tp = await self.set_take_profit(symbol, float(take_profit))
                                    if not ok_tp: logger.warning("Could not set take profit via trading_stop")
                                return order
                        except Exception as e2:
                            logger.error(f"Retry {i} (no in-order TP/SL) failed: {e2}")

                logger.error(f"Order placement failed: {emsg}")
                return None

        except Exception as e:
            logger.error(f"place_order_with_sl_tp fatal error: {e}")
            return None
            
        
    async def fetch_realized_pnl(self, symbol: str, since_ms: int | None = None) -> float | None:
        """
        Compute realized PnL for `symbol` using fills since `since_ms`.
        Works by summing SELL proceeds - BUY costs - fees.
        Returns None if it cannot determine a flat-close (i.e., net position ≠ 0 over the window).
        """
        try:
            # category=linear for USDT perps on Bybit v5
            params = {"category": "linear"}
            trades = await self.exchange.fetch_my_trades(symbol, since=since_ms, limit=200, params=params)
        except Exception as e:
            # keep silent in production; log if you like:
            # logger.debug(f"fetch_realized_pnl: fetch_my_trades failed: {e}")
            return None

        if not trades:
            return None

        buy_qty = 0.0
        sell_qty = 0.0
        buy_cost = 0.0
        sell_cost = 0.0
        total_fee = 0.0

        for t in trades:
            side = (t.get("side") or "").lower()
            amt = float(t.get("amount") or 0.0)
            cost = float(t.get("cost") or 0.0)

            if side == "buy":
                buy_qty += amt
                buy_cost += cost
            elif side == "sell":
                sell_qty += amt
                sell_cost += cost

            fee_obj = t.get("fee") or {}
            try:
                total_fee += float(fee_obj.get("cost") or 0.0)
            except Exception:
                pass

        # Only report realized PnL if the window nets to flat (position fully closed)
        if abs(sell_qty - buy_qty) > 1e-9:
            return None

        realized = sell_cost - buy_cost - total_fee
        return float(realized)
   
                
