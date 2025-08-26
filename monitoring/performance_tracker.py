"""
Performance Tracking and Analytics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import json

from utils.logger import get_logger

logger = get_logger(__name__)

class PerformanceTracker:
    def __init__(self):
        self.trades = []
        self.daily_stats = {}
        self.start_balance = 0
        self.current_balance = 0
        self.peak_balance = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
    def update(self, positions: Dict, balance: float):
        """Update performance metrics"""
        if self.start_balance == 0:
            self.start_balance = balance
            self.peak_balance = balance
        
        self.current_balance = balance
        
        if balance > self.peak_balance:
            self.peak_balance = balance
        
        # Calculate current metrics
        self.calculate_metrics()
    
    def add_trade(self, trade: Dict):
        """Add a completed trade"""
        trade['timestamp'] = datetime.now()
        self.trades.append(trade)
        self.total_trades += 1
        
        if trade.get('pnl', 0) > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        logger.info(f"Trade recorded: {trade}")
    
    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self.trades:
            return {}
        
        df = pd.DataFrame(self.trades)
        
        metrics = {
            'total_return': (self.current_balance - self.start_balance) / self.start_balance if self.start_balance > 0 else 0,
            'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
            'max_drawdown': (self.peak_balance - self.current_balance) / self.peak_balance if self.peak_balance > 0 else 0,
            'sharpe_ratio': self.calculate_sharpe_ratio(df),
            'sortino_ratio': self.calculate_sortino_ratio(df),
            'profit_factor': self.calculate_profit_factor(df),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades
        }
        
        return metrics
    
    def calculate_sharpe_ratio(self, df: pd.DataFrame) -> float:
        """Calculate Sharpe ratio"""
        if df.empty or 'pnl' not in df.columns:
            return 0
        
        returns = df['pnl'] / self.start_balance
        if len(returns) < 2:
            return 0
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return 0
        
        # Annualized Sharpe ratio (assuming daily returns)
        return (mean_return / std_return) * np.sqrt(365)
    
    def calculate_sortino_ratio(self, df: pd.DataFrame) -> float:
        """Calculate Sortino ratio"""
        if df.empty or 'pnl' not in df.columns:
            return 0
        
        returns = df['pnl'] / self.start_balance
        if len(returns) < 2:
            return 0
        
        mean_return = returns.mean()
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return 0
        
        downside_std = downside_returns.std()
        
        if downside_std == 0:
            return 0
        
        # Annualized Sortino ratio
        return (mean_return / downside_std) * np.sqrt(365)
    
    def calculate_profit_factor(self, df: pd.DataFrame) -> float:
        """Calculate profit factor"""
        if df.empty or 'pnl' not in df.columns:
            return 0
        
        gross_profit = df[df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(df[df['pnl'] < 0]['pnl'].sum())
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0
        
        return gross_profit / gross_loss
    
    def save_report(self):
        """Save performance report"""
        metrics = self.calculate_metrics()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'trades': len(self.trades),
            'start_balance': self.start_balance,
            'current_balance': self.current_balance
        }
        
        # Save to file
        with open(f"logs/performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report saved: {metrics}")
