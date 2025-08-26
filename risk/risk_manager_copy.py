## 📁 File 9: `risk/risk_manager.py` - Risk Management System
```python
"""
Advanced Risk Management System
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
        """Check if trailing stop should be updated"""
        entry_price = position.get('entry_price', current_price)
        current_stop = position.get('stop_loss', 0)
        
        if position.get('side') == 'long':
            # For long positions
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct > 0.02:  # 2% profit
                new_stop = current_price * 0.98  # Trail at 2% below current
                return new_stop > current_stop
        else:
            # For short positions
            profit_pct = (entry_price - current_price) / entry_price
            if profit_pct > 0.02:  # 2% profit
                new_stop = current_price * 1.02  # Trail at 2% above current
                return new_stop < current_stop
        
        return False
    
    def calculate_trailing_stop(self, position: Dict, current_price: float) -> float:
        """Calculate new trailing stop price"""
        if position.get('side') == 'long':
            return current_price * 0.98  # 2% below current price
        else:
            return current_price * 1.02  # 2% above current price