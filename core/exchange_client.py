import os, time, asyncio, json
from typing import Any, Dict, Optional, List
from loguru import logger
import ccxt.async_support as ccxt  # async ccxt

class ExchangeClient:
    """
    Thin async wrapper around ccxt.bybit with Bybit v5-friendly params.
    Includes robust order placement with tpslMode, price-band clamp, and positionIdx retry.
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

        # Balance (safe log)
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

    async def fetch_order_book(self, symbol: str, limit: int = 5):
        try:
            return await self.exchange.fetch_order_book(symbol, limit=limit)
        except Exception as e:
            logger.error(f"fetch_order_book error {symbol}: {e}")
            return {"bids": [], "asks": []}

    # Unified wrappers
    async def create_market_order(self, symbol, side, amount, params=None):
        return await self.exchange.create_order(symbol, "market", side, amount, None, params or {})

    async def create_limit_order(self, symbol, side, amount, price, params=None):
        return await self.exchange.create_order(symbol, "limit", side, amount, price, params or {})

    async def set_stop_loss(self, symbol: str, price: float) -> bool:
        """Bybit v5 trading_stop via raw endpoint (ccxt may not have a unified helper)."""
        try:
            market = self.exchange.market(symbol)
            req = {
                "category": "linear",
                "symbol": market["id"],
                "stopLoss": str(round(float(price), 6)),
                "slTriggerBy": "LastPrice",
            }
            # Prefer v5 endpoint name; fallback to older naming if ccxt version differs
            if hasattr(self.exchange, "privatePostV5PositionTradingStop"):
                await self.exchange.privatePostV5PositionTradingStop(req)
            elif hasattr(self.exchange, "privatePostContractV3PrivatePositionTradingStop"):
                await self.exchange.privatePostContractV3PrivatePositionTradingStop(req)
            else:
                logger.warning("No trading_stop endpoint found in ccxt; skipping set_stop_loss")
                return False
            return True
        except Exception as e:
            logger.error(f"set_stop_loss failed: {e}")
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

    # --- The robust order placement you asked for (with all the fixes) ---
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

            # Direction sanity
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
                params["stopLoss"] = str(round(float(stop_loss), 6))
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
                    # Maker-first for non-scalp strategies
                    p2 = dict(p)
                    if signal_strategy not in ("scalping", "advanced_scalping"):
                        # try post only (maker)
                        p2["postOnly"] = True
                        try:
                            return await self.create_limit_order(symbol, side_l, quantity, use_price, params=p2)
                        except Exception:
                            # fall back to normal limit if post-only rejected
                            p2.pop("postOnly", None)
                    return await self.create_limit_order(symbol, side_l, quantity, use_price, params=p2)

            safe_limit_price: Optional[float] = None

            # First attempt (no positionIdx)
            try:
                order = await _submit(params)
                if order:
                    self.orders[order.get("id", f"{symbol}-{time.time()}")] = order
                    logger.info(f"✅ Order placed successfully: {order.get('id','<no-id>')}")
                    return order
            except Exception as first_err:
                emsg = str(first_err)
                lower = emsg.lower()

                # Parse retCode if present
                ret_code = None
                try:
                    jtxt = emsg.split("bybit", 1)[1].strip()
                    jobj = json.loads(jtxt)
                    ret_code = int(jobj.get("retCode"))
                except Exception:
                    pass

                # A) position mode mismatch -> retry with Hedge positionIdx
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

                # B) Price band violation -> clamp to top-of-book and retry LIMIT
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

                # C) Strip TP/SL and place clean order, reuse safe_limit_price if set
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
