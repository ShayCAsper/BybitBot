"""
Enhanced Bot Manager with Trade Cooldowns and Position Limits
Prevents overtrading and manages trade frequency properly
"""

import asyncio
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import defaultdict
import logging

from core.exchange_client import BybitClient
from strategies.strategy_manager import StrategyManager
from risk.risk_manager import RiskManager
from ml.ml_predictor import MLPredictor
from utils.logger import get_logger
from monitoring.performance_tracker import PerformanceTracker

logger = get_logger(__name__)

class BotManager:
    
    async def startup(self, exchange):
        """
        Call once after creating the exchange client.
        Forces One-Way position mode for linear perps (if enabled),
        so we don't hit 'position idx not match position mode'.
        """
        self.exchange = exchange

        if self.force_one_way:
            try:
                # False = One-Way (not hedged). category='linear' for USDT perps.
                await self.exchange.set_position_mode(False, None, params={'category': 'linear'})
                logger.info("Set Bybit position mode to One-Way for linear perps")
            except Exception as e:
                logger.warning(f"Could not set position mode globally: {e}")

        self.position_mode_ready = True
    
    def __init__(self, config: Dict[str, Any]):
        
        self.config = config
        self.exchange = None
        self.strategy_manager = None
        self.risk_manager = None
        self.ml_predictor = None
        self.performance_tracker = None

        # --- NEW: position-mode control ---
        # Set to 1 (true) to force One-Way on startup; 0 to skip forcing
        self.force_one_way = bool(int(os.getenv('FORCE_ONE_WAY_MODE', '1')))
        self.position_mode_ready = False
        # -----------------------------------

        # State management
        self.active = False
        self.positions = {}
        self.balance = 0
        self.initial_balance = 0

        # Performance tracking
        self.trade_history = []
        self.consecutive_losses = 0
        self.daily_trades = 0

        # TRADE COOLDOWN MANAGEMENT
        self.last_trade_time = {}  # Per symbol
        self.last_signal_time = {}  # Per symbol
        self.trade_cooldowns = {
            'scalping': 60,        # 1 minute between scalping trades on same coin
            'momentum': 180,       # 3 minutes between momentum trades
            'mean_reversion': 300, # 5 minutes between mean reversion trades
            'default': 120         # 2 minutes default
        }

        # POSITION LIMITS
        self.max_positions_per_coin = 1  # Only 1 position per coin
        self.max_total_positions = int(os.getenv('MAX_POSITIONS', '3'))
        self.min_candles_between_trades = int(os.getenv('MIN_CANDLES_BETWEEN_TRADES', '5'))

        # Signal filtering
        self.recent_signals = defaultdict(list)  # Track recent signals to avoid duplicates
        self.signal_expiry = 30  # Signals expire after 30 seconds

        # Trading enablement
        self.trading_enabled = self._check_trading_enabled()

        # Dynamic intervals
        self.base_interval = int(os.getenv('LOOP_INTERVAL', '5'))
        self.current_interval = self.base_interval

        # Strategy-specific intervals
        self.strategy_intervals = {
            'scalping': 3,
            'momentum': 5,
            'mean_reversion': 10,
            'ml': 10,
            'aggressive': 5,
            'conservative': 15,
        }
        
        # --- Performance & risk knobs ---
        self.daily_loss_cap_pct = float(os.getenv('DAILY_LOSS_CAP_PCT', '1.5'))  # pause if -1.5%
        self.max_daily_trades   = int(os.getenv('MAX_DAILY_TRADES', '20'))
        self.min_rr_momentum    = float(os.getenv('MIN_RR_MOMENTUM', '1.2'))
        self.min_rr_meanrev     = float(os.getenv('MIN_RR_MEANREV', '1.2'))
        self.min_rr_scalp       = float(os.getenv('MIN_RR_SCALP', '1.05'))

        self.max_consec_losses  = int(os.getenv('MAX_CONSEC_LOSSES', '3'))
        self.loss_cooldown_min  = int(os.getenv('LOSS_COOLDOWN_MIN', '45'))

        # UTC session filter (inclusive start, exclusive end, e.g., "12-20")
        self.session_utc_window = os.getenv('SESSION_HOURS_UTC', '12-20')

        # Dynamic loop bounds (seconds)
        self.min_loop_interval  = int(os.getenv('MIN_LOOP_INTERVAL', '3'))
        self.max_loop_interval  = int(os.getenv('MAX_LOOP_INTERVAL', '10'))

    def _check_trading_enabled(self) -> bool:
        """Check if trading is enabled"""
        from_env = os.getenv('ENABLE_TRADING', 'False')
        enabled = from_env.lower() in ['true', '1', 'yes', 'on'] if isinstance(from_env, str) else bool(from_env)
        logger.info(f"🔧 Trading Enabled: {enabled}")
        return enabled
    
    def _can_trade_symbol(self, symbol: str, strategy: str = 'default') -> bool:
        """Check if we can trade this symbol based on cooldowns and limits"""
        
        # Check if we already have a position in this symbol
        if symbol in self.positions:
            logger.debug(f"Already have position in {symbol}, skipping new trade")
            return False
        
        # Check total positions limit
        if len(self.positions) >= self.max_total_positions:
            logger.debug(f"Max positions reached ({self.max_total_positions}), skipping trade")
            return False
        
        # Check cooldown for this symbol
        if symbol in self.last_trade_time:
            cooldown_period = self.trade_cooldowns.get(strategy, self.trade_cooldowns['default'])
            time_since_last_trade = (datetime.now() - self.last_trade_time[symbol]).total_seconds()
            
            if time_since_last_trade < cooldown_period:
                remaining = cooldown_period - time_since_last_trade
                logger.debug(f"Cooldown active for {symbol}: {remaining:.0f}s remaining")
                return False
        
        return True
    
    def _is_duplicate_signal(self, signal: Dict) -> bool:
        """Check if this is a duplicate signal"""
        symbol = signal['symbol']
        side = signal['side']
        strategy = signal.get('strategy', 'unknown')
        
        # Create signal key
        signal_key = f"{symbol}_{side}_{strategy}"
        
        # Clean old signals
        current_time = datetime.now()
        self.recent_signals[signal_key] = [
            s for s in self.recent_signals[signal_key]
            if (current_time - s).total_seconds() < self.signal_expiry
        ]
        
        # Check if we've seen this signal recently
        if self.recent_signals[signal_key]:
            logger.debug(f"Duplicate signal filtered: {signal_key}")
            return True
        
        # Add to recent signals
        self.recent_signals[signal_key].append(current_time)
        return False
    
    def _calculate_dynamic_interval(self) -> int:
        """Calculate optimal loop interval based on conditions"""
        
        # Base interval from active strategies
        active_strategies = self.strategy_manager.active_strategies if self.strategy_manager else []
        min_interval = self.base_interval
        
        for strategy in active_strategies:
            if strategy in self.strategy_intervals:
                min_interval = min(min_interval, self.strategy_intervals[strategy])
        
        # If we have recent trades, slow down a bit
        recent_trades = sum(
            1 for t in self.last_trade_time.values()
            if (datetime.now() - t).total_seconds() < 60
        )
        
        if recent_trades >= 2:
            # We've traded recently, slow down
            min_interval = max(min_interval, 10)
            logger.debug(f"Recent trades detected, using slower interval: {min_interval}s")
        
        # If we have max positions, slow down
        if len(self.positions) >= self.max_total_positions:
            min_interval = max(min_interval, 15)
            logger.debug(f"Max positions reached, using slower interval: {min_interval}s")
        
        # Never go below 2 seconds to avoid API rate limits
        min_interval = max(min_interval, 2)
        
        return min_interval
    
    async def initialize(self):
        """Initialize all components"""
        logger.info("🚀 Initializing Bot Manager with Trade Controls...")
        
        try:
            # Initialize exchange
            self.exchange = BybitClient(self.config['exchange'])
            await self.exchange.connect()
            
            # Get initial balance
            self.balance = await self.exchange.get_balance()
            self.initial_balance = self.balance
            logger.info(f"💰 Initial balance: ${self.balance:.2f}")
            
            # Initialize components
            self.risk_manager = RiskManager(self.config['risk'])
            self.ml_predictor = MLPredictor(self.config.get('ml', {}))
            await self.ml_predictor.load_models()
            
            self.strategy_manager = StrategyManager(
                self.config['strategies'],
                self.exchange,
                self.risk_manager,
                self.ml_predictor
            )
            
            self.performance_tracker = PerformanceTracker()
            
            # Log configuration
            logger.info("⚙️ Trade Control Settings:")
            logger.info(f"  Max positions per coin: {self.max_positions_per_coin}")
            logger.info(f"  Max total positions: {self.max_total_positions}")
            logger.info(f"  Min candles between trades: {self.min_candles_between_trades}")
            logger.info(f"  Cooldown periods: {self.trade_cooldowns}")
            
            logger.info("✅ Bot Manager initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise
    
    async def start(self):
        """Start the trading bot"""
        self.active = True
        logger.info(f"🚀 Bot Started! Trading: {'ENABLED ✅' if self.trading_enabled else 'DISABLED (Simulation Mode) ⚠️'}")
        
        tasks = [
            asyncio.create_task(self.trading_loop()),
            asyncio.create_task(self.monitor_positions()),
            asyncio.create_task(self.performance_monitor()),
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Critical error: {e}")
        finally:
            await self.shutdown()
    
    async def trading_loop(self):
        """Main trading loop with proper trade management"""
        await asyncio.sleep(5)  # Initial delay
        
        loop_count = 0
        while self.active:
            loop_count += 1
            loop_start = datetime.now()
            
            try:
                # Calculate dynamic interval
                self.current_interval = self._calculate_dynamic_interval()
                
                # Only log every 10th loop to reduce spam
                if loop_count % 10 == 0:
                    logger.info(f"🔄 Trading Loop #{loop_count} (Interval: {self.current_interval}s)")
                
                # Get market data
                market_data = await self.exchange.get_market_data()
                
                if not market_data:
                    logger.warning("No market data available")
                    await asyncio.sleep(self.current_interval)
                    continue
                
                # Get ML predictions (optional)
                predictions = {}
                try:
                    predictions = await self.ml_predictor.predict(market_data)
                except:
                    pass
                
                # Generate trading signals
                signals = await self.strategy_manager.generate_signals(market_data, predictions)
                
                if signals:
                    # Filter signals through trade controls
                    valid_signals = self._filter_valid_signals(signals)
                    
                    if valid_signals:
                        logger.info(f"📊 {len(valid_signals)} valid signals out of {len(signals)} total")
                        
                        if self.trading_enabled:
                            # Execute only the best signal
                            best_signal = self._select_best_signal(valid_signals)
                            if best_signal:
                                success = await self.execute_trade(best_signal)
                                if success:
                                    logger.info(f"✅ Trade executed for {best_signal['symbol']}")
                                    # Update last trade time
                                    self.last_trade_time[best_signal['symbol']] = datetime.now()
                        else:
                            logger.info("⚠️ TRADING DISABLED - Valid signals not executed")
                            for signal in valid_signals[:3]:  # Show max 3 signals
                                self._log_signal(signal)
                
                # Calculate sleep time
                loop_duration = (datetime.now() - loop_start).total_seconds()
                sleep_time = max(0.1, self.current_interval - loop_duration)
                
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(self.current_interval)
    
    def _filter_valid_signals(self, signals: List[Dict]) -> List[Dict]:
        """Filter signals through trade controls"""
        valid_signals = []
        
        for signal in signals:
            symbol = signal['symbol']
            strategy = signal.get('strategy', 'default')
            
            # Check if duplicate
            if self._is_duplicate_signal(signal):
                continue
            
            # Check if we can trade this symbol
            if not self._can_trade_symbol(symbol, strategy):
                continue
            
            # Check minimum confidence
            if signal.get('confidence', 0) < 0.5:
                continue
            
            valid_signals.append(signal)
        
        return valid_signals
    
    def _select_best_signal(self, signals: List[Dict]) -> Optional[Dict]:
        """Select the best signal from valid signals"""
        if not signals:
            return None
        
        # Sort by confidence and strategy weight
        signals.sort(key=lambda x: x.get('confidence', 0) * x.get('weight', 1), reverse=True)
        
        # Return the best signal
        return signals[0]
    
    async def execute_trade(self, signal: Dict) -> bool:
        """Execute trade with proper SL/TP calculation based on side"""
        try:
            symbol = signal['symbol']
            side = signal['side']
  
            # Final check before execution
            if not self._can_trade_symbol(symbol, signal.get('strategy', 'default')):
                logger.warning(f"Trade blocked by final check for {symbol}")
                return False
            
            # Get current price
            current_price = await self.exchange.get_price(symbol)
            if current_price == 0:
                logger.error(f"Could not get price for {symbol}")
                return False
            
            # Calculate position size
            position_size = self._calculate_position_size(signal, current_price)
            
            # Ensure minimum size
            min_sizes = {'BTC': 0.001, 'ETH': 0.01, 'SOL': 0.1}
            coin = symbol.split('/')[0]
            min_size = min_sizes.get(coin, 0.001)
            position_size = max(position_size, min_size)
            
            logger.info(f"🎯 Executing: {side.upper()} {position_size:.4f} {symbol} @ ${current_price:.2f}")
            
            # CRITICAL: Calculate SL/TP correctly based on side
            # The percentages are the DISTANCE from current price
            
            if signal.get('strategy') == 'scalping':
                sl_distance = 0.005  # 0.5% distance
                tp_distance = 0.005  # 0.5% distance
            elif signal.get('strategy') == 'momentum':
                sl_distance = 0.015  # 1.5% distance
                tp_distance = 0.025  # 2.5% distance
            else:
                sl_distance = 0.02   # 2% distance
                tp_distance = 0.03   # 3% distance
            
            # Apply SL/TP based on trade direction
            if side.lower() in ['buy', 'long']:
                # BUY/LONG positions:
                stop_loss = current_price * (1 - sl_distance)   # SL below entry
                take_profit = current_price * (1 + tp_distance)  # TP above entry
                logger.debug(f"BUY: Entry=${current_price:.2f}, SL=${stop_loss:.2f} (-{sl_distance*100:.1f}%), TP=${take_profit:.2f} (+{tp_distance*100:.1f}%)")
                
            elif side.lower() in ['sell', 'short']:
                # SELL/SHORT positions:
                stop_loss = current_price * (1 + sl_distance)   # SL above entry
                take_profit = current_price * (1 - tp_distance)  # TP below entry
                logger.debug(f"SELL: Entry=${current_price:.2f}, SL=${stop_loss:.2f} (+{sl_distance*100:.1f}%), TP=${take_profit:.2f} (-{tp_distance*100:.1f}%)")
            else:
                logger.error(f"Unknown side: {side}")
                return False
            
            # Override with signal's SL/TP if provided and valid
            if 'stop_loss' in signal and signal['stop_loss'] > 0:
                # Validate the signal's stop loss
                if side.lower() in ['buy', 'long'] and signal['stop_loss'] < current_price:
                    stop_loss = signal['stop_loss']
                elif side.lower() in ['sell', 'short'] and signal['stop_loss'] > current_price:
                    stop_loss = signal['stop_loss']
                else:
                    logger.warning(f"Invalid stop loss in signal: ${signal['stop_loss']:.2f}, using calculated: ${stop_loss:.2f}")
            
            if 'take_profit' in signal and signal['take_profit'] > 0:
                # Validate the signal's take profit
                if side.lower() in ['buy', 'long'] and signal['take_profit'] > current_price:
                    take_profit = signal['take_profit']
                elif side.lower() in ['sell', 'short'] and signal['take_profit'] < current_price:
                    take_profit = signal['take_profit']
                else:
                    logger.warning(f"Invalid take profit in signal: ${signal['take_profit']:.2f}, using calculated: ${take_profit:.2f}")
            
            # Place the order with validated SL/TP
            order = await self.exchange.place_order_with_sl_tp(
                symbol=symbol,
                side=side,
                quantity=position_size,
                order_type='market',
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if order:
                # Track position
                self.positions[symbol] = {
                    'order_id': order['id'],
                    'side': side,
                    'entry_price': current_price,
                    'quantity': position_size,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'timestamp': datetime.now(),
                    'strategy': signal.get('strategy', 'unknown')
                }
                
                self.daily_trades += 1
                
                logger.info(f"✅ Order placed: {order['id']}")
                logger.info(f"   Strategy: {signal.get('strategy', 'unknown')}")
                logger.info(f"   Direction: {side.upper()}")
                logger.info(f"   Entry: ${current_price:.2f}")
                logger.info(f"   Stop Loss: ${stop_loss:.2f} ({sl_distance*100:.1f}% risk)")
                logger.info(f"   Take Profit: ${take_profit:.2f} ({tp_distance*100:.1f}% reward)")
                logger.info(f"   Risk/Reward: 1:{tp_distance/sl_distance:.1f}")
                
                # Log cooldown
                cooldown = self.trade_cooldowns.get(signal.get('strategy', 'default'), 120)
                logger.info(f"   Next trade on {symbol} allowed in: {cooldown}s")
                
                return True
            else:
                logger.error(f"Order placement returned None for {symbol}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to execute trade: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    def _calculate_position_size(self, signal: Dict, current_price: float) -> float:
        """Calculate position size with limits"""
        base_value = float(os.getenv('BASE_POSITION_VALUE', 100))
        
        # Adjust for strategy
        strategy_multipliers = {
            'scalping': 0.5,     # Smaller for scalping
            'momentum': 1.0,     # Normal for momentum
            'mean_reversion': 1.2, # Slightly larger for mean reversion
        }
        
        multiplier = strategy_multipliers.get(signal.get('strategy'), 1.0)
        base_value *= multiplier
        
        # Adjust for confidence
        confidence = signal.get('confidence', 0.5)
        adjusted_value = base_value * (0.5 + confidence * 0.5)
        
        # Convert to coin quantity
        position_size = adjusted_value / current_price
        
        # Apply maximum limits
        max_size = self.balance * 0.02 / current_price  # Max 2% of balance per trade
        position_size = min(position_size, max_size)
        
        return position_size
    
    def _log_signal(self, signal: Dict):
        """Log signal details"""
        logger.info(
            f"📊 Signal: {signal.get('strategy', 'unknown')} | "
            f"{signal['side'].upper()} {signal['symbol']} | "
            f"Confidence: {signal.get('confidence', 0):.2%}"
        )
    
    async def monitor_positions(self):
        """Monitor open positions with appropriate frequency"""
        await asyncio.sleep(10)
        
        while self.active:
            try:
                if not self.positions:
                    await asyncio.sleep(10)
                    continue
                
                # Get actual positions from exchange
                exchange_positions = await self.exchange.get_positions()
                
                for symbol, local_pos in list(self.positions.items()):
                    # Find matching exchange position
                    exchange_pos = next(
                        (p for p in exchange_positions if p['symbol'] == symbol),
                        None
                    )
                    
                    if not exchange_pos or exchange_pos.get('contracts', 0) == 0:
                        # Position closed
                        logger.info(f"📊 Position closed: {symbol}")
                        
                        # Calculate how long position was held
                        duration = datetime.now() - local_pos['timestamp']
                        logger.info(f"   Duration: {duration.total_seconds():.0f}s")
                        
                        # Remove from tracking
                        del self.positions[symbol]
                        continue
                    
                    # Update trailing stop only if significantly profitable
                    current_price = await self.exchange.get_price(symbol)
                    entry_price = local_pos['entry_price']
                    
                    if local_pos['side'] == 'buy':
                        profit_pct = (current_price - entry_price) / entry_price
                        
                        # Only trail if profit > 1%
                        if profit_pct > 0.01:
                            new_stop = current_price * 0.99
                            if new_stop > local_pos.get('stop_loss', 0) * 1.001:  # Only update if significantly better
                                success = await self.exchange.update_stop_loss(symbol, new_stop)
                                if success:
                                    local_pos['stop_loss'] = new_stop
                                    logger.info(f"📈 Trailing stop updated: {symbol} @ ${new_stop:.2f}")
                
                # Check positions every 5 seconds
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error monitoring positions: {e}")
                await asyncio.sleep(10)
    
    async def performance_monitor(self):
        """Monitor and log performance"""
        await asyncio.sleep(60)
        
        while self.active:
            try:
                # Update balance
                self.balance = await self.exchange.get_balance()
                
                # Calculate metrics
                pnl = self.balance - self.initial_balance
                pnl_pct = (pnl / self.initial_balance * 100) if self.initial_balance > 0 else 0
                
                # Count recent trades
                recent_trades = sum(
                    1 for t in self.last_trade_time.values()
                    if (datetime.now() - t).total_seconds() < 3600
                )
                
                logger.info("📊 === PERFORMANCE UPDATE ===")
                logger.info(f"Balance: ${self.balance:.2f}")
                logger.info(f"P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
                logger.info(f"Open Positions: {len(self.positions)}/{self.max_total_positions}")
                logger.info(f"Daily Trades: {self.daily_trades}")
                logger.info(f"Recent Trades (1h): {recent_trades}")
                logger.info(f"Current Interval: {self.current_interval}s")
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")
                await asyncio.sleep(300)
    
    async def shutdown(self):
        """Gracefully shutdown"""
        logger.info("🛑 Shutting down bot...")
        self.active = False
        
        if self.exchange:
            await self.exchange.disconnect()
        
        logger.info("✅ Bot shutdown complete")
