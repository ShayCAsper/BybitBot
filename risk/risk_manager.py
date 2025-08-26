"""
Advanced Risk Management System with Configurable Trailing Stops
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from utils.logger import get_logger

logger = get_logger(__name__)

class RiskManager:
    def __init__(self, config: Dict):
        self.config = config
        self.max_position_size = config.get('max_position_size', 0.1)
        self.max_positions = config.get('max_positions', 5)
        self.max_daily_loss = config.get('max_daily_loss', 0.05)
        self.max_drawdown = config.get('max_drawdown', 0.15)
        self.risk_per_trade = config.get('risk_per_trade', 0.02)
        self.correlation_limit = config.get('correlation_limit', 0.7)
        
        # TRAILING STOP CONFIGURATION
        self.trailing_stop_config = config.get('trailing_stop', {})
        self.enable_trailing = self.trailing_stop_config.get('enabled', True)
        self.trailing_type = self.trailing_stop_config.get('type', 'percentage')  # percentage, atr, fixed
        self.trailing_distance = self.trailing_stop_config.get('distance', 0.02)  # 2% default
        self.trailing_activation = self.trailing_stop_config.get('activation_profit', 0.01)  # Activate after 1% profit
        self.trailing_step = self.trailing_stop_config.get('step', 0.001)  # Minimum step to update
        
        # Advanced trailing stop features
        self.breakeven_enabled = self.trailing_stop_config.get('breakeven_enabled', True)
        self.breakeven_trigger = self.trailing_stop_config.get('breakeven_trigger', 0.005)  # 0.5% profit
        self.partial_close_enabled = self.trailing_stop_config.get('partial_close_enabled', False)
        self.partial_close_levels = self.trailing_stop_config.get('partial_close_levels', [
            {'profit': 0.02, 'close_percentage': 0.25},  # Close 25% at 2% profit
            {'profit': 0.04, 'close_percentage': 0.25},  # Close another 25% at 4% profit
        ])
        
        # Track position highs for trailing
        self.position_highs = {}
        self.position_trail_active = {}
        
        # Track daily P&L
        self.daily_pnl = 0
        self.daily_trades = 0
        self.peak_balance = 0
        self.current_drawdown = 0
    
    def check_risk_limits(self, signal: Dict, positions: Dict) -> bool:
        """Check if signal passes risk limits"""
        
        # Check max positions
        if len(positions) >= self.max_positions:
            logger.warning(f"Max positions reached: {len(positions)}/{self.max_positions}")
            return False
        
        # Check position size
        if signal.get('quantity', 0) > self.max_position_size:
            logger.warning(f"Position size too large: {signal['quantity']}")
            return False
        
        # Check daily loss limit
        if self.daily_pnl < -self.max_daily_loss:
            logger.warning(f"Daily loss limit reached: {self.daily_pnl}")
            return False
        
        # Check correlation with existing positions
        if not self.check_correlation(signal, positions):
            logger.warning("Position correlation too high")
            return False
        
        return True
    
    def check_correlation(self, signal: Dict, positions: Dict) -> bool:
        """Check correlation with existing positions"""
        # Simplified correlation check
        # In production, calculate actual correlation between assets
        same_direction_positions = sum(
            1 for pos in positions.values()
            if pos.get('side') == signal.get('side')
        )
        
        if same_direction_positions >= 3:
            return False
        
        return True
    
    def calculate_position_size_by_signal(self, signal: Dict, balance: float) -> float:
        """Calculate position size based on signal strength"""
        base_size = 0.001  # Minimum BTC size
        
        # Adjust based on signal strength
        strength_multipliers = {
            'strong': 2.0,   # 3 strategies agree
            'medium': 1.5,   # 2 strategies agree
            'weak': 1.0      # Single strategy
        }
        
        multiplier = strength_multipliers.get(signal.get('strength', 'weak'), 1.0)
        
        # Adjust based on confidence
        confidence = signal.get('confidence', 0.5)
        confidence_multiplier = 0.5 + confidence  # 0.5x to 1.5x
        
        # Calculate final size
        position_size = base_size * multiplier * confidence_multiplier
        
        # Apply limits
        max_size = balance * 0.05 / signal.get('price', 1)  # Max 5% of balance
        position_size = min(position_size, max_size)
        position_size = max(position_size, base_size)  # At least minimum
        
        # Round to exchange precision
        return round(position_size, 4)
    
    def calculate_position_size(
        self, 
        signal: Dict, 
        account_balance: float, 
        current_price: float
    ) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        
        # Get win rate and risk/reward from historical data
        win_rate = self.config.get('historical_win_rate', 0.6)
        avg_win = self.config.get('avg_win', 0.03)
        avg_loss = self.config.get('avg_loss', 0.02)
        
        # Kelly Criterion
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        # Apply risk per trade limit
        risk_amount = account_balance * self.risk_per_trade
        stop_distance = abs(current_price - signal.get('stop_loss', current_price * 0.98))
        
        if stop_distance > 0:
            position_size = risk_amount / stop_distance
        else:
            position_size = account_balance * 0.01
        
        # Apply Kelly fraction
        position_size *= kelly_fraction
        
        # Apply maximum position size
        position_size = min(position_size, account_balance * self.max_position_size)
        
        return position_size
    
    def calculate_portfolio_risk(self, positions: Dict, balance: float) -> Dict:
        """Calculate overall portfolio risk metrics"""
        
        total_exposure = sum(
            pos.get('contracts', 0) * pos.get('entry_price', 0)
            for pos in positions.values()
        )
        
        # Calculate Value at Risk (VaR)
        var_95 = self.calculate_var(positions, confidence=0.95)
        var_99 = self.calculate_var(positions, confidence=0.99)
        
        # Update drawdown
        if balance > self.peak_balance:
            self.peak_balance = balance
        self.current_drawdown = (self.peak_balance - balance) / self.peak_balance if self.peak_balance > 0 else 0
        
        return {
            'total_exposure': total_exposure,
            'exposure_ratio': total_exposure / balance if balance > 0 else 0,
            'var_95': var_95,
            'var_99': var_99,
            'drawdown': self.current_drawdown,
            'daily_pnl': self.daily_pnl,
            'position_count': len(positions),
            'risk_score': self.calculate_risk_score(positions, balance)
        }
    
    def calculate_var(self, positions: Dict, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        # Simplified VaR calculation
        # In production, use historical data and more sophisticated models
        
        if not positions:
            return 0
        
        position_values = [
            pos.get('contracts', 0) * pos.get('entry_price', 0)
            for pos in positions.values()
        ]
        
        if not position_values:
            return 0
        
        # Assume normal distribution with historical volatility
        volatility = self.config.get('historical_volatility', 0.02)
        z_score = np.abs(np.percentile(np.random.standard_normal(10000), (1 - confidence) * 100))
        
        portfolio_value = sum(position_values)
        var = portfolio_value * volatility * z_score
        
        return var
    
    def calculate_risk_score(self, positions: Dict, balance: float) -> float:
        """Calculate overall risk score (0-100)"""
        score = 0
        
        # Exposure risk (0-30 points)
        total_exposure = sum(
            pos.get('contracts', 0) * pos.get('entry_price', 0)
            for pos in positions.values()
        )
        exposure_ratio = total_exposure / balance if balance > 0 else 0
        score += min(exposure_ratio * 100, 30)
        
        # Drawdown risk (0-30 points)
        score += self.current_drawdown * 200
        
        # Position count risk (0-20 points)
        score += (len(positions) / self.max_positions) * 20
        
        # Daily loss risk (0-20 points)
        daily_loss_ratio = abs(self.daily_pnl) / self.max_daily_loss if self.daily_pnl < 0 else 0
        score += daily_loss_ratio * 20
        
        return min(score, 100)
   
    def should_update_trailing_stop(self, position: Dict, current_price: float) -> bool:
        """Enhanced trailing stop logic with multiple strategies"""
        if not self.enable_trailing:
            return False
        
        position_id = position.get('id', position.get('symbol'))
        entry_price = position.get('entry_price', current_price)
        current_stop = position.get('stop_loss', 0)
        side = position.get('side')
        
        # Calculate profit percentage
        if side == 'long' or side == 'buy':
            profit_pct = (current_price - entry_price) / entry_price
            is_profitable = current_price > entry_price
        else:  # short/sell
            profit_pct = (entry_price - current_price) / entry_price
            is_profitable = current_price < entry_price
        
        # Check if trailing stop should be activated
        if not self.position_trail_active.get(position_id, False):
            if profit_pct >= self.trailing_activation:
                self.position_trail_active[position_id] = True
                logger.info(f"Trailing stop activated for {position_id} at {profit_pct:.2%} profit")
            elif self.breakeven_enabled and profit_pct >= self.breakeven_trigger:
                # Move stop to breakeven
                return True
            else:
                return False
        
        # Track position high/low
        if side == 'long' or side == 'buy':
            if position_id not in self.position_highs or current_price > self.position_highs[position_id]:
                self.position_highs[position_id] = current_price
                return True
        else:  # short/sell
            if position_id not in self.position_highs or current_price < self.position_highs[position_id]:
                self.position_highs[position_id] = current_price
                return True
        
        return False
    
    def calculate_trailing_stop(self, position: Dict, current_price: float, market_data: Optional[Dict] = None) -> float:
        """Calculate new trailing stop price with multiple methods"""
        position_id = position.get('id', position.get('symbol'))
        entry_price = position.get('entry_price', current_price)
        side = position.get('side')
        
        # Calculate profit percentage
        if side == 'long' or side == 'buy':
            profit_pct = (current_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - current_price) / entry_price
        
        # Breakeven stop
        if self.breakeven_enabled and profit_pct >= self.breakeven_trigger and profit_pct < self.trailing_activation:
            if side == 'long' or side == 'buy':
                return entry_price * 1.001  # Slightly above entry for longs
            else:
                return entry_price * 0.999  # Slightly below entry for shorts
        
        # Calculate trailing stop based on type
        if self.trailing_type == 'percentage':
            return self._calculate_percentage_trailing_stop(position, current_price)
        elif self.trailing_type == 'atr':
            return self._calculate_atr_trailing_stop(position, current_price, market_data)
        elif self.trailing_type == 'fixed':
            return self._calculate_fixed_trailing_stop(position, current_price)
        elif self.trailing_type == 'dynamic':
            return self._calculate_dynamic_trailing_stop(position, current_price, market_data)
        else:
            return self._calculate_percentage_trailing_stop(position, current_price)
    
    def _calculate_percentage_trailing_stop(self, position: Dict, current_price: float) -> float:
        """Percentage-based trailing stop"""
        side = position.get('side')
        position_id = position.get('id', position.get('symbol'))
        
        if side == 'long' or side == 'buy':
            # For longs, trail below the highest price
            highest = self.position_highs.get(position_id, current_price)
            return highest * (1 - self.trailing_distance)
        else:
            # For shorts, trail above the lowest price
            lowest = self.position_highs.get(position_id, current_price)
            return lowest * (1 + self.trailing_distance)
    
    def _calculate_atr_trailing_stop(self, position: Dict, current_price: float, market_data: Optional[Dict] = None) -> float:
        """ATR-based trailing stop (more adaptive to volatility)"""
        side = position.get('side')
        symbol = position.get('symbol')
        
        # Get ATR from market data if available
        atr = 0.02  # Default 2% if no ATR data
        if market_data and symbol in market_data:
            atr_value = market_data[symbol].get('atr', None)
            if atr_value:
                atr = atr_value / current_price  # Convert to percentage
        
        # Use ATR multiplier from config
        atr_multiplier = self.trailing_stop_config.get('atr_multiplier', 2.0)
        trail_distance = atr * atr_multiplier
        
        if side == 'long' or side == 'buy':
            return current_price * (1 - trail_distance)
        else:
            return current_price * (1 + trail_distance)
    
    def _calculate_fixed_trailing_stop(self, position: Dict, current_price: float) -> float:
        """Fixed dollar amount trailing stop"""
        side = position.get('side')
        fixed_amount = self.trailing_stop_config.get('fixed_amount', 10)  # $10 default
        
        if side == 'long' or side == 'buy':
            return current_price - fixed_amount
        else:
            return current_price + fixed_amount
    
    def _calculate_dynamic_trailing_stop(self, position: Dict, current_price: float, market_data: Optional[Dict] = None) -> float:
        """Dynamic trailing stop based on profit level"""
        side = position.get('side')
        entry_price = position.get('entry_price', current_price)
        
        # Calculate profit percentage
        if side == 'long' or side == 'buy':
            profit_pct = (current_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - current_price) / entry_price
        
        # Dynamic distance based on profit
        if profit_pct < 0.02:  # Less than 2% profit
            trail_distance = 0.015  # Tight 1.5% trail
        elif profit_pct < 0.05:  # 2-5% profit
            trail_distance = 0.02  # 2% trail
        elif profit_pct < 0.10:  # 5-10% profit
            trail_distance = 0.03  # 3% trail
        else:  # More than 10% profit
            trail_distance = 0.04  # Wider 4% trail
        
        if side == 'long' or side == 'buy':
            return current_price * (1 - trail_distance)
        else:
            return current_price * (1 + trail_distance)
    
    def should_partial_close(self, position: Dict, current_price: float) -> Optional[Dict]:
        """Check if position should be partially closed"""
        if not self.partial_close_enabled:
            return None
        
        entry_price = position.get('entry_price', current_price)
        side = position.get('side')
        position_id = position.get('id', position.get('symbol'))
        
        # Calculate profit percentage
        if side == 'long' or side == 'buy':
            profit_pct = (current_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - current_price) / entry_price
        
        # Check partial close levels
        for level in self.partial_close_levels:
            level_profit = level['profit']
            close_pct = level['close_percentage']
            
            # Check if this level has been executed
            level_key = f"{position_id}_{level_profit}"
            if profit_pct >= level_profit and not hasattr(self, f'partial_closed_{level_key}'):
                setattr(self, f'partial_closed_{level_key}', True)
                return {
                    'action': 'partial_close',
                    'percentage': close_pct,
                    'reason': f'Partial close at {level_profit:.1%} profit'
                }
        
        return None
    
    # ... (rest of the existing RiskManager methods remain the same)