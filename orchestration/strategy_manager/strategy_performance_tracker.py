import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import time
from loguru import logger

@dataclass
class StrategyMetrics:
    strategy_id: str
    returns: List[float]
    signals: List[Dict]
    trades: List[Dict]
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    avg_holding_period: float
    success_rate: float

class StrategyPerformanceTracker:
    """
    Comprehensive tracking and analysis of individual strategy performance
    """
    
    def __init__(self, lookback_days: int = 30):
        self.lookback_days = lookback_days
        self.strategy_data = defaultdict(lambda: {
            'signals': deque(maxlen=1000),
            'trades': deque(maxlen=1000),
            'returns': deque(maxlen=1000),
            'execution_quality': deque(maxlen=100)
        })
        
    def record_strategy_signal(self, strategy_id: str, signal_data: Dict):
        """Record a strategy signal"""
        signal_data['timestamp'] = time.time()
        self.strategy_data[strategy_id]['signals'].append(signal_data)
        
    def record_strategy_trade(self, strategy_id: str, trade_data: Dict):
        """Record a completed trade"""
        trade_data['timestamp'] = time.time()
        self.strategy_data[strategy_id]['trades'].append(trade_data)
        
        # Calculate return
        if 'pnl' in trade_data and 'capital' in trade_data:
            trade_return = trade_data['pnl'] / trade_data['capital']
            self.strategy_data[strategy_id]['returns'].append(trade_return)
    
    def calculate_strategy_metrics(self, strategy_id: str) -> Optional[StrategyMetrics]:
        """Calculate comprehensive metrics for a strategy"""
        try:
            if strategy_id not in self.strategy_data:
                return None
            
            data = self.strategy_data[strategy_id]
            
            # Filter recent data
            cutoff_time = time.time() - (self.lookback_days * 24 * 3600)
            
            recent_trades = [t for t in data['trades'] if t['timestamp'] >= cutoff_time]
            recent_returns = [r for r in data['returns'] if len(recent_trades) > 0]
            recent_signals = [s for s in data['signals'] if s['timestamp'] >= cutoff_time]
            
            if not recent_trades:
                return None
            
            # Calculate metrics
            total_return = sum(recent_returns) if recent_returns else 0.0
            winning_trades = [t for t in recent_trades if t.get('pnl', 0) > 0]
            win_rate = len(winning_trades) / len(recent_trades) if recent_trades else 0.0
            
            # Sharpe ratio
            if recent_returns and len(recent_returns) > 1:
                mean_return = np.mean(recent_returns)
                std_return = np.std(recent_returns)
                sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
            else:
                sharpe_ratio = 0.0
            
            # Max drawdown
            if recent_returns:
                cumulative = np.cumprod(1 + np.array(recent_returns))
                peak = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - peak) / peak
                max_drawdown = abs(np.min(drawdown))
            else:
                max_drawdown = 0.0
            
            # Average holding period
            holding_periods = [t.get('holding_period', 0) for t in recent_trades]
            avg_holding_period = np.mean(holding_periods) if holding_periods else 0.0
            
            # Success rate (signals that led to profitable trades)
            success_rate = win_rate  # Simplified for now
            
            return StrategyMetrics(
                strategy_id=strategy_id,
                returns=recent_returns,
                signals=list(recent_signals),
                trades=list(recent_trades),
                win_rate=win_rate,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                total_return=total_return,
                avg_holding_period=avg_holding_period,
                success_rate=success_rate
            )
            
        except Exception as e:
            logger.error(f"❌ Strategy metrics calculation error: {e}")
            return None
    
    def get_strategy_comparison(self) -> Dict:
        """Compare performance across all strategies"""
        try:
            comparison = {}
            
            for strategy_id in self.strategy_data.keys():
                metrics = self.calculate_strategy_metrics(strategy_id)
                if metrics:
                    comparison[strategy_id] = {
                        'total_return': metrics.total_return,
                        'sharpe_ratio': metrics.sharpe_ratio,
                        'win_rate': metrics.win_rate,
                        'max_drawdown': metrics.max_drawdown,
                        'total_trades': len(metrics.trades)
                    }
            
            return comparison
            
        except Exception as e:
            logger.error(f"❌ Strategy comparison error: {e}")
            return {}
