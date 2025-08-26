"""
Bybit Exchange Client - Complete Fixed Version
Handles all exchange interactions with proper error handling
"""

import ccxt.pro as ccxt
import asyncio
from typing import Dict, List, Optional, Any
from decimal import Decimal
import time

from utils.logger import get_logger

logger = get_logger(__name__)

class BybitClient:
    def __init__(self, config: Dict[str, Any]):
      
        # Advanced Scalping specific coins
        advanced_scalping_coins = config.get('advanced_scalping_coins', 
            ['BTC', 'ETH', 'SOL'])  # Default high-liquidity coins for scalping
        
        # Regular trading coins
        regular_coins = config.get('trading_coins', 
            ['BTC', 'ETH', 'SOL', 'DOGE', 'MATIC'])
        
        # Combine both lists (remove duplicates)
        all_coins = list(set(advanced_scalping_coins + regular_coins))
        
        self.config = config
        self.exchange = None
        self.symbols = config.get('symbols', [f"{coin}/USDT:USDT" for coin in all_coins])
        self.orderbooks = {}
        self.tickers = {}
        self.positions = {}
        self.orders = {}
        
        logger.info(f"Initialized with coins: {all_coins}")
        logger.info(f"Advanced Scalping coins: {advanced_scalping_coins}")
        
    async def connect(self):
        """Connect to Bybit exchange"""
        try:
            api_key = self.config.get('api_key')
            api_secret = self.config.get('api_secret')
            testnet = self.config.get('testnet', True)
            
            logger.info(f"Connecting to Bybit {'Testnet' if testnet else 'Mainnet'}...")
            
            # Initialize exchange with proper testnet configuration
            if testnet:
                self.exchange = ccxt.bybit({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'swap',
                        'adjustForTimeDifference': True,
                        'recvWindow': 10000,
                    },
                    'urls': {
                        'api': {
                            'public': 'https://api-testnet.bybit.com',
                            'private': 'https://api-testnet.bybit.com',
                        }
                    }
                })
            else:
                self.exchange = ccxt.bybit({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'swap',
                        'adjustForTimeDifference': True,
                        'recvWindow': 10000,
                    }
                })
            
            self.exchange.set_sandbox_mode(testnet)
            markets = await self.exchange.load_markets()
            logger.info(f"Markets loaded: {len(markets)} markets available")
            
            try:
                balance = await self.exchange.fetch_balance()
                usdt_balance = balance.get('USDT', {}).get('free', 0)
                logger.info(f"✅ Connected to Bybit {'Testnet' if testnet else 'Mainnet'}")
                logger.info(f"Account balance: {usdt_balance} USDT")
            except Exception as balance_error:
                logger.warning(f"Could not fetch balance: {balance_error}")
                logger.info(f"✅ Connected to Bybit {'Testnet' if testnet else 'Mainnet'}")
            
            await self.initialize_leverage()

        except Exception as e:
            logger.error(f"Failed to connect to Bybit: {e}")
            if self.exchange:
                await self.exchange.close()
            raise
    
                
    async def get_balance(self) -> float:
        """Get account balance"""
        try:
            balance = await self.exchange.fetch_balance()
            usdt_balance = 0
            
            if 'USDT' in balance:
                usdt_balance = balance['USDT'].get('free', 0)
            
            if 'info' in balance and 'result' in balance['info']:
                try:
                    if 'list' in balance['info']['result']:
                        for account in balance['info']['result']['list']:
                            if 'coin' in account:
                                for coin in account['coin']:
                                    if coin.get('coin') == 'USDT':
                                        usdt_balance = float(coin.get('walletBalance', 0))
                except:
                    pass
            
            return float(usdt_balance)
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0
    
    async def get_market_data(self) -> Dict:
        """Get comprehensive market data"""
        market_data = {}
        
        for symbol in self.symbols:
            try:
                if symbol not in self.exchange.markets:
                    logger.warning(f"Symbol {symbol} not found in markets")
                    continue
                
                ticker = await self.exchange.fetch_ticker(symbol)
                ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe='1m', limit=100)
                orderbook = await self.exchange.fetch_order_book(symbol, limit=20)
                
                market_data[symbol] = {
                    'ticker': ticker,
                    'ohlcv': ohlcv,
                    'orderbook': orderbook,
                    'timestamp': time.time()
                }
                
            except Exception as e:
                logger.error(f"Failed to get market data for {symbol}: {e}")
        
        return market_data
    
    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = 'market',
        reduce_only: bool = False
    ) -> Optional[Dict]:
        """Place an order on Bybit"""
        try:
            params = {}
            if reduce_only:
                params['reduceOnly'] = True
            
            if order_type == 'market':
                order = await self.exchange.create_market_order(
                    symbol, side, quantity, params=params
                )
            else:
                order = await self.exchange.create_limit_order(
                    symbol, side, quantity, price, params=params
                )
            
            self.orders[order['id']] = order
            logger.info(f"Order placed: {order['id']} - {side} {quantity} {symbol}")
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None
            
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order"""
        try:
            await self.exchange.cancel_order(order_id, symbol)
            if order_id in self.orders:
                del self.orders[order_id]
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
            
    async def place_order_with_sl_tp(
            self,
            symbol: str,
            side: str,
            quantity: float,
            price: Optional[float] = None,
            order_type: str = 'market',
            stop_loss: Optional[float] = None,
            take_profit: Optional[float] = None,
            reduce_only: bool = False
        ) -> Optional[Dict]:
        """
        Place an order with optional stop loss and take profit on Bybit v5.

        Flow:
          1) Try without positionIdx (One-Way friendly) + attach TP/SL (tpslMode='Full').
          2) If 'position idx not match position mode' -> retry with Hedge positionIdx: 1=BUY/long, 2=SELL/short.
          3) If params still fail (or TP/SL cause issues), place CLEAN order (no TP/SL), then set SL/TP via trading_stop.
          4) If price hits Bybit price band (retCode 30208), clamp to top-of-book and reuse that clamped price
             for all subsequent retries (and make TP/SL valid vs that price).
        """
        try:
            current_price = await self.get_price(symbol)
            if not current_price or current_price <= 0:
                self.log_error("Invalid current price received for {symbol}: {price}", symbol=symbol, price=current_price)
                return None

            logger.info(f"Current market price for {symbol}: ${current_price:,.2f}")

            side_l = side.lower()
            order_type_l = (order_type or 'market').lower()
            if order_type_l not in ('market', 'limit'):
                logger.warning(f"Unsupported order_type '{order_type}', defaulting to 'market'")
                order_type_l = 'market'

            # --- Ensure SL/TP are on correct side of price hint (current price for now) ---
            if side_l in ('buy', 'long'):
                if stop_loss is not None and stop_loss >= current_price:
                    old = stop_loss; stop_loss = current_price * 0.98
                    logger.warning(f"BUY: Adjusted SL {old:.2f} -> {stop_loss:.2f} (must be < price)")
                if take_profit is not None and take_profit <= current_price:
                    old = take_profit; take_profit = current_price * 1.03
                    logger.warning(f"BUY: Adjusted TP {old:.2f} -> {take_profit:.2f} (must be > price)")
                if order_type_l == 'limit' and price is not None and price > current_price:
                    old = price; price = current_price * 0.999
                    logger.warning(f"BUY: Adjusted limit {old:.2f} -> {price:.2f} (must be <= market)")
            else:  # sell / short
                if stop_loss is not None and stop_loss <= current_price:
                    old = stop_loss; stop_loss = current_price * 1.02
                    logger.warning(f"SELL: Adjusted SL {old:.2f} -> {stop_loss:.2f} (must be > price)")
                if take_profit is not None and take_profit >= current_price:
                    old = take_profit; take_profit = current_price * 0.97
                    logger.warning(f"SELL: Adjusted TP {old:.2f} -> {take_profit:.2f} (must be < price)")
                if order_type_l == 'limit' and price is not None and price < current_price:
                    old = price; price = current_price * 1.001
                    logger.warning(f"SELL: Adjusted limit {old:.2f} -> {price:.2f} (must be >= market)")

            # --- Build params (first attempt: NO positionIdx, category=linear) ---
            params: Dict[str, Any] = {'category': 'linear'}
            if reduce_only:
                params['reduceOnly'] = True

            has_sl = stop_loss is not None
            has_tp = take_profit is not None
            if has_sl or has_tp:
                params['tpslMode'] = 'Full'
            if has_sl:
                params['stopLoss'] = str(round(float(stop_loss), 6))
                params['slTriggerBy'] = 'LastPrice'
                params['slOrderType'] = 'Market'
            if has_tp:
                params['takeProfit'] = str(round(float(take_profit), 6))
                params['tpTriggerBy'] = 'LastPrice'
                params['tpOrderType'] = 'Market'

            logger.info(
                f"Placing {order_type_l.upper()} {side_l.upper()} order:\n"
                f"  Symbol: {symbol}\n"
                f"  Qty: {quantity}\n"
                f"  Price: {('MKT' if price is None else f'${price:.6f}')}\n"
                f"  SL: {('—' if stop_loss is None else f'${stop_loss:.6f}')}\n"
                f"  TP: {('—' if take_profit is None else f'${take_profit:.6f}')}"
            )

            async def _submit(p: Dict[str, Any], order_price: Optional[float] = None) -> Optional[Dict]:
                if order_type_l == 'market':
                    return await self.exchange.create_market_order(symbol, side_l, quantity, params=p)
                else:
                    use_price = order_price if order_price is not None else (price if price is not None else current_price)
                    return await self.exchange.create_limit_order(symbol, side_l, quantity, use_price, params=p)

            safe_limit_price: Optional[float] = None  # set if we hit price band

            # ---------------------- First attempt ----------------------
            try:
                order = await _submit(params)
                if order:
                    self.orders[order.get('id', f"{symbol}-{time.time()}")] = order
                    logger.info(f"✅ Order placed successfully: {order.get('id','<no-id>')}")
                    return order

            except Exception as first_err:
                import json, asyncio
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

                # --------- A) position mode mismatch -> retry with Hedge positionIdx ----------
                if 'position idx not match position mode' in lower:
                    logger.warning("Bybit: positionIdx mismatch. Retrying in Hedge mode with positionIdx...")
                    hedge_params = dict(params)
                    hedge_params['positionIdx'] = 1 if side_l in ('buy', 'long') else 2
                    try:
                        order = await _submit(hedge_params)
                        if order:
                            self.orders[order.get('id', f"{symbol}-{time.time()}")] = order
                            logger.info(f"✅ Order placed on retry (Hedge positionIdx): {order.get('id','<no-id>')}")
                            return order
                    except Exception as second_err:
                        logger.error(f"Retry with positionIdx failed: {second_err}")
                        # fallthrough

                # --------- B) price band (30208 or message) -> clamp to top-of-book ----------
                if ret_code == 30208 or 'maximum buying price' in lower or 'minimum selling price' in lower:
                    logger.warning(f"Price band hit: {emsg}. Fetching top-of-book to clamp price and retry as LIMIT...")
                    try:
                        ob = await self.exchange.fetch_order_book(symbol, limit=1)
                        best_bid = ob['bids'][0][0] if ob.get('bids') else current_price
                        best_ask = ob['asks'][0][0] if ob.get('asks') else current_price

                        if side_l in ('buy','long'):
                            safe_limit_price = best_ask * 1.0002
                            # ensure TP > safe_limit_price; SL < safe_limit_price
                            if has_tp and take_profit is not None and take_profit <= safe_limit_price:
                                take_profit = safe_limit_price * 1.002
                            if has_sl and stop_loss is not None and stop_loss >= safe_limit_price:
                                stop_loss = safe_limit_price * 0.995
                        else:
                            safe_limit_price = best_bid * 0.9998
                            # ensure TP < safe_limit_price; SL > safe_limit_price
                            if has_tp and take_profit is not None and take_profit >= safe_limit_price:
                                take_profit = safe_limit_price * 0.998
                            if has_sl and stop_loss is not None and stop_loss <= safe_limit_price:
                                stop_loss = safe_limit_price * 1.005

                        # precision
                        try:
                            safe_limit_price = float(self.exchange.price_to_precision(symbol, safe_limit_price))
                        except Exception:
                            pass

                        # refresh TP/SL in params if still attaching them
                        params_clamped = dict(params)
                        if has_tp and take_profit is not None:
                            try:
                                take_profit = float(self.exchange.price_to_precision(symbol, take_profit))
                            except Exception:
                                pass
                            params_clamped['takeProfit'] = str(round(float(take_profit), 6))
                        if has_sl and stop_loss is not None:
                            try:
                                stop_loss = float(self.exchange.price_to_precision(symbol, stop_loss))
                            except Exception:
                                pass
                            params_clamped['stopLoss'] = str(round(float(stop_loss), 6))

                        logger.warning(f"Clamped price -> retrying LIMIT {side_l.upper()} at {safe_limit_price}")
                        # Force LIMIT retry with clamped price and (possibly) adjusted TP/SL
                        # Temporarily treat as limit for this retry
                        saved_type = order_type_l
                        order_type_l = 'limit'
                        try:
                            order = await _submit(params_clamped, order_price=safe_limit_price)
                            if order:
                                self.orders[order.get('id', f"{symbol}-{time.time()}")] = order
                                logger.info(f"✅ Order placed after price clamp: {order.get('id','<no-id>')}")
                                return order
                        finally:
                            order_type_l = saved_type  # restore

                    except Exception as clamp_err:
                        logger.error(f"Retry with clamped price failed: {clamp_err}")
                        # fallthrough

                # --------- C) strip TP/SL and place clean order, reusing safe_limit_price if set ----------
                must_strip_tpsl = any(x in lower for x in (
                    'tpslmode', 'tpordertype', 'slordertype',
                    'tpordertype can not have a value when', 'param error'
                )) or 'position idx not match position mode' in lower or ret_code == 30208

                if must_strip_tpsl:
                    logger.warning(f"Retrying order WITHOUT in-order TP/SL, will set via trading_stop... ({emsg})")

                    def _strip_tpsl(p: Dict[str, Any]) -> Dict[str, Any]:
                        keys = ('tpslMode','stopLoss','slTriggerBy','slOrderType','takeProfit','tpTriggerBy','tpOrderType')
                        return {k: v for k, v in p.items() if k not in keys}

                    v1 = _strip_tpsl(params)  # no positionIdx
                    v2 = _strip_tpsl(params); v2['positionIdx'] = 1 if side_l in ('buy','long') else 2

                    # If we hit price band, force LIMIT with the safe price for these retries
                    for i, ptry in enumerate((v1, v2), 1):
                        try:
                            if safe_limit_price is not None:
                                saved_type = order_type_l
                                order_type_l = 'limit'
                                try:
                                    order = await _submit(ptry, order_price=safe_limit_price)
                                finally:
                                    order_type_l = saved_type
                            else:
                                order = await _submit(ptry)

                            if order:
                                self.orders[order.get('id', f"{symbol}-{time.time()}")] = order
                                logger.info(f"✅ Order placed on retry {i} (no in-order TP/SL): {order.get('id','<no-id>')}")
                                await asyncio.sleep(1.0)
                                if has_sl and stop_loss is not None:
                                    ok_sl = await self.set_stop_loss(symbol, float(stop_loss))
                                    if not ok_sl:
                                        logger.warning("Could not set stop loss via trading_stop")
                                if has_tp and take_profit is not None:
                                    ok_tp = await self.set_take_profit(symbol, float(take_profit))
                                    if not ok_tp:
                                        logger.warning("Could not set take profit via trading_stop")
                                return order
                        except Exception as e2:
                            logger.error(f"Retry {i} (no in-order TP/SL) failed: {e2}")

                logger.error(f"Order placement failed: {emsg}")
                return None

        except Exception as e:
            logger.error(f"place_order_with_sl_tp fatal error: {e}")
            return None

    async def set_take_profit(self, symbol: str, price: float) -> bool:
        """Set take profit for a position - FIXED for v5"""
        try:
            positions = await self.exchange.fetch_positions([symbol])
            
            if not positions or len(positions) == 0:
                logger.warning(f"No position found for {symbol}")
                return False
            
            position = positions[0]
            
            if position.get('contracts', 0) == 0:
                logger.warning(f"No open position for {symbol}")
                return False
            
            bybit_symbol = symbol.replace('/', '').replace(':USDT', '')
            
            try:
                params = {
                    'category': 'linear',
                    'symbol': bybit_symbol,
                    'takeProfit': str(round(price, 2)),
                    'tpTriggerBy': 'LastPrice',
                    'tpSlMode': 'Full',  # Add this for v5
                    'positionIdx': 0
                }
                
                response = await self.exchange.private_post_v5_position_trading_stop(params)
                
                if response.get('retCode') in [0, '0', 34040]:
                    logger.info(f"✅ Take profit set for {symbol} at ${price:.2f}")
                    return True
                else:
                    logger.error(f"Failed to set take profit: {response}")
                    return False
                    
            except Exception as e:
                if "not modified" in str(e).lower() or "34040" in str(e):
                    logger.debug(f"Take profit already set for {symbol}")
                    return True
                logger.error(f"Failed to set take profit: {e}")
                return False
                    
        except Exception as e:
            logger.error(f"Failed to set take profit: {e}")
            return False
    
    async def update_stop_loss(self, symbol: str, new_price: float) -> bool:
        """Update stop loss (for trailing stops)"""
        return await self.set_stop_loss(symbol, new_price)
    
    async def set_position_sl_tp(self, symbol: str, stop_loss: float = None, take_profit: float = None) -> bool:
        """Set both stop loss and take profit for a position"""
        try:
            positions = await self.exchange.fetch_positions([symbol])
            
            if not positions or len(positions) == 0:
                logger.warning(f"No position found for {symbol}")
                return False
            
            position = positions[0]
            if position.get('contracts', 0) == 0:
                logger.warning(f"No open position for {symbol}")
                return False
            
            bybit_symbol = symbol.replace('/', '').replace(':USDT', '')
            
            params = {
                'category': 'linear',
                'symbol': bybit_symbol,
                'positionIdx': 0
            }
            
            if stop_loss:
                params['stopLoss'] = str(round(stop_loss, 2))
                params['slTriggerBy'] = 'LastPrice'
            
            if take_profit:
                params['takeProfit'] = str(round(take_profit, 2))
                params['tpTriggerBy'] = 'LastPrice'
            
            response = await self.exchange.private_post_v5_position_trading_stop(params)
            
            if response.get('retCode') == 0 or response.get('retCode') == '0':
                if stop_loss:
                    logger.info(f"✅ Stop loss set at ${stop_loss:.2f}")
                if take_profit:
                    logger.info(f"✅ Take profit set at ${take_profit:.2f}")
                return True
            else:
                logger.error(f"Failed to set SL/TP: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to set position SL/TP: {e}")
            return False
    
    async def get_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return float(ticker['last'])
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return 0.0
    
    async def close_position(self, symbol: str) -> bool:
        """Close a position"""
        try:
            positions = await self.exchange.fetch_positions([symbol])
            if positions and len(positions) > 0:
                position = positions[0]
                if position['contracts'] > 0:
                    side = 'sell' if position['side'] == 'long' else 'buy'
                    await self.place_order(
                        symbol=symbol,
                        side=side,
                        quantity=position['contracts'],
                        order_type='market',
                        reduce_only=True
                    )
                    logger.info(f"Position closed for {symbol}")
                    return True
        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
            return False
    
    async def get_positions(self) -> List[Dict]:
        """Get all open positions"""
        try:
            positions = await self.exchange.fetch_positions()
            return [p for p in positions if p['contracts'] > 0]
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order"""
        try:
            await self.exchange.cancel_order(order_id, symbol)
            if order_id in self.orders:
                del self.orders[order_id]
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol"""
        try:
            positions = await self.exchange.fetch_positions([symbol])
            if positions and len(positions) > 0:
                current_leverage = positions[0].get('leverage', 0)
                if current_leverage == leverage:
                    logger.info(f"Leverage already set to {leverage}x for {symbol}")
                    return True
            
            try:
                result = await self.exchange.set_leverage(leverage, symbol)
                logger.info(f"Leverage set to {leverage}x for {symbol}")
                return True
            except Exception as e:
                if "leverage not modified" in str(e).lower():
                    logger.info(f"Leverage already at {leverage}x for {symbol}")
                    return True
                raise e
                
        except Exception as e:
            logger.warning(f"Leverage setting skipped: {e}")
            return False
    
    async def initialize_leverage(self):
        """Initialize leverage for all trading symbols"""
        default_leverage = self.config.get('default_leverage', 10)
        
        for symbol in self.symbols:
            symbol_key = symbol.split('/')[0] + '_LEVERAGE'
            leverage = self.config.get(symbol_key.lower(), default_leverage)
            
            await self.set_leverage(symbol, leverage)
            logger.info(f"Initialized {symbol} with {leverage}x leverage")
    
    async def disconnect(self):
        """Disconnect from exchange"""
        if self.exchange:
            try:
                await self.exchange.close()
                logger.info("Disconnected from Bybit")
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
    
    # Fetch methods for compatibility
    async def fetch_ticker(self, symbol: str):
        """Fetch ticker data for a symbol"""
        try:
            return await self.exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"Failed to fetch ticker for {symbol}: {e}")
            return {}
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 100):
        """Fetch OHLCV data"""
        try:
            return await self.exchange.fetch_ohlcv(symbol, timeframe, limit)
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            return []
    
    async def fetch_order_book(self, symbol: str, limit: int = 20):
        """Fetch order book data"""
        try:
            return await self.exchange.fetch_order_book(symbol, limit)
        except Exception as e:
            logger.error(f"Failed to fetch order book for {symbol}: {e}")
            return {'bids': [], 'asks': []}