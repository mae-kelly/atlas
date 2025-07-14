import asyncio
import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from loguru import logger
import empyrical
from datetime import datetime, timedelta
try:
    import quantlib as ql
    QUANTLIB_AVAILABLE = True
except ImportError:
    QUANTLIB_AVAILABLE = False
    logger.warning("QuantLib not available - some advanced features disabled")
@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    entry_time: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    weight: float
    duration_hours: float
    cost_basis: float
    fees_paid: float
@dataclass
class PortfolioSnapshot:
    timestamp: float
    total_value: float
    cash: float
    positions: Dict[str, Position]
    total_unrealized_pnl: float
    total_realized_pnl: float
    daily_return: float
    cumulative_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    average_trade_duration: float
    number_of_positions: int
@dataclass
class PerformanceMetrics:
    period_start: float
    period_end: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    number_of_trades: int
    average_trade_duration: float
    turnover: float
    beta: Optional[float] = None
    alpha: Optional[float] = None
    information_ratio: Optional[float] = None
class PortfolioManagementEngine:
    """
    Comprehensive portfolio management system with real-time tracking,
    performance analysis, and optimization
    """
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 benchmark_symbol: str = "BTC",
                 performance_window: int = 252):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.benchmark_symbol = benchmark_symbol
        self.performance_window = performance_window
        self.positions: Dict[str, Position] = {}
        self.cash = initial_capital
        self.portfolio_history: deque = deque(maxlen=10000)
        self.trades_history: List[Dict] = []
        self.daily_returns: deque = deque(maxlen=performance_window)
        self.portfolio_values: deque = deque(maxlen=performance_window)
        self.benchmark_returns: deque = deque(maxlen=performance_window)
        self.current_metrics: Optional[PerformanceMetrics] = None
        self.last_metrics_update: float = 0
        self.metrics_update_interval: float = 3600
        self.risk_alerts: List[Dict] = []
        self.risk_thresholds = {
            'max_drawdown': 0.15,      # 15% max drawdown
            'daily_var': 0.05,         # 5% daily VaR
            'concentration': 0.3,       # 30% max single position
            'leverage': 2.0            # 2x max leverage
        }
        self.attribution_data: Dict = defaultdict(list)
        self.portfolio_update_handlers: List = []
        self.performance_alert_handlers: List = []
        logger.info("📊 Portfolio Management Engine initialized")
    def add_portfolio_update_handler(self, handler):
        """Add callback for portfolio updates"""
        self.portfolio_update_handlers.append(handler)
    def add_performance_alert_handler(self, handler):
        """Add callback for performance alerts"""
        self.performance_alert_handlers.append(handler)
    async def update_position(self, symbol: str, quantity: float, price: float, 
                            trade_type: str = "unknown", strategy_id: str = None):
        """
        Update or create position
        """
        try:
            current_time = time.time()
            if symbol in self.positions:
                position = self.positions[symbol]
                if (position.quantity > 0 and quantity < 0) or (position.quantity < 0 and quantity > 0):
                    trade_quantity = min(abs(quantity), abs(position.quantity))
                    realized_pnl = trade_quantity * (price - position.entry_price) * (1 if position.quantity > 0 else -1)
                    position.realized_pnl += realized_pnl
                    self._record_trade(symbol, trade_quantity, position.entry_price, price, 
                                     position.entry_time, current_time, realized_pnl, strategy_id)
                if abs(position.quantity + quantity) < 1e-8:
                    del self.positions[symbol]
                    logger.info(f"📤 Position closed: {symbol}")
                else:
                    old_quantity = position.quantity
                    new_quantity = position.quantity + quantity
                    if (old_quantity > 0 and quantity > 0) or (old_quantity < 0 and quantity < 0):
                        total_cost = (old_quantity * position.entry_price) + (quantity * price)
                        position.entry_price = total_cost / new_quantity
                        position.cost_basis = abs(new_quantity * position.entry_price)
                    position.quantity = new_quantity
                    position.current_price = price
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=price,
                    current_price=price,
                    entry_time=current_time,
                    market_value=abs(quantity * price),
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    weight=0.0,
                    duration_hours=0.0,
                    cost_basis=abs(quantity * price),
                    fees_paid=0.0
                )
                logger.info(f"📥 New position: {symbol} - {quantity} @ {price}")
            self.cash -= quantity * price
            await self._update_portfolio_metrics()
            await self._check_risk_alerts()
            await self._notify_portfolio_update()
        except Exception as e:
            logger.error(f"❌ Position update error: {e}")
    async def update_market_prices(self, price_data: Dict[str, float]):
        """
        Update current market prices for all positions
        """
        try:
            portfolio_value_changed = False
            for symbol, price in price_data.items():
                if symbol in self.positions:
                    position = self.positions[symbol]
                    old_price = position.current_price
                    position.current_price = price
                    position.market_value = abs(position.quantity * price)
                    position.unrealized_pnl = position.quantity * (price - position.entry_price)
                    position.duration_hours = (time.time() - position.entry_time) / 3600
                    portfolio_value_changed = True
            if portfolio_value_changed:
                await self._update_portfolio_metrics()
                if time.time() - self.last_metrics_update > self.metrics_update_interval:
                    await self._calculate_performance_metrics()
        except Exception as e:
            logger.error(f"❌ Price update error: {e}")
    async def _update_portfolio_metrics(self):
        """Update real-time portfolio metrics"""
        try:
            current_time = time.time()
            total_position_value = sum(pos.market_value for pos in self.positions.values())
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
            total_portfolio_value = self.cash + total_position_value
            for position in self.positions.values():
                position.weight = position.market_value / total_portfolio_value if total_portfolio_value > 0 else 0
            if self.portfolio_values:
                previous_value = self.portfolio_values[-1]
                daily_return = (total_portfolio_value - previous_value) / previous_value
            else:
                daily_return = (total_portfolio_value - self.initial_capital) / self.initial_capital
            self.daily_returns.append(daily_return)
            self.portfolio_values.append(total_portfolio_value)
            cumulative_return = (total_portfolio_value - self.initial_capital) / self.initial_capital
            sharpe_ratio = self._calculate_sharpe_ratio()
            max_drawdown = self._calculate_max_drawdown()
            win_rate = self._calculate_win_rate()
            avg_trade_duration = self._calculate_average_trade_duration()
            snapshot = PortfolioSnapshot(
                timestamp=current_time,
                total_value=total_portfolio_value,
                cash=self.cash,
                positions=self.positions.copy(),
                total_unrealized_pnl=total_unrealized_pnl,
                total_realized_pnl=total_realized_pnl,
                daily_return=daily_return,
                cumulative_return=cumulative_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                average_trade_duration=avg_trade_duration,
                number_of_positions=len(self.positions)
            )
            self.portfolio_history.append(snapshot)
            self.current_capital = total_portfolio_value
        except Exception as e:
            logger.error(f"❌ Portfolio metrics update error: {e}")
    def _record_trade(self, symbol: str, quantity: float, entry_price: float, 
                     exit_price: float, entry_time: float, exit_time: float, 
                     pnl: float, strategy_id: str = None):
        """Record completed trade"""
        trade = {
            'symbol': symbol,
            'quantity': quantity,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'duration_hours': (exit_time - entry_time) / 3600,
            'pnl': pnl,
            'pnl_percentage': pnl / (quantity * entry_price) if entry_price > 0 else 0,
            'strategy_id': strategy_id,
            'trade_type': 'win' if pnl > 0 else 'loss',
            'timestamp': exit_time
        }
        self.trades_history.append(trade)
        logger.info(f"📝 Trade recorded: {symbol} - P&L: {pnl:.2f} ({trade['pnl_percentage']:.2%})")
    async def _calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        try:
            if len(self.daily_returns) < 30:
                return
            current_time = time.time()
            period_start = current_time - (len(self.daily_returns) * 24 * 3600)
            returns_array = np.array(list(self.daily_returns))
            total_return = (self.current_capital - self.initial_capital) / self.initial_capital
            try:
                annualized_return = empyrical.annual_return(returns_array)
                volatility = empyrical.annual_volatility(returns_array)
                sharpe_ratio = empyrical.sharpe_ratio(returns_array)
                sortino_ratio = empyrical.sortino_ratio(returns_array)
                max_drawdown = empyrical.max_drawdown(returns_array)
            except Exception as e:
                logger.warning(f"Empyrical calculation error: {e}, using fallbacks")
                annualized_return = np.mean(returns_array) * 252
                volatility = np.std(returns_array) * np.sqrt(252)
                sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
                sortino_ratio = self._calculate_sortino_fallback(returns_array)
                max_drawdown = self._calculate_max_drawdown()
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            win_rate = self._calculate_win_rate()
            profit_factor = self._calculate_profit_factor()
            avg_win, avg_loss = self._calculate_avg_win_loss()
            largest_win, largest_loss = self._calculate_largest_win_loss()
            avg_trade_duration = self._calculate_average_trade_duration()
            turnover = self._calculate_turnover()
            beta, alpha, information_ratio = self._calculate_market_metrics()
            self.current_metrics = PerformanceMetrics(
                period_start=period_start,
                period_end=current_time,
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_drawdown,
                max_drawdown_duration=0,
                win_rate=win_rate,
                profit_factor=profit_factor,
                average_win=avg_win,
                average_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                number_of_trades=len(self.trades_history),
                average_trade_duration=avg_trade_duration,
                turnover=turnover,
                beta=beta,
                alpha=alpha,
                information_ratio=information_ratio
            )
            self.last_metrics_update = current_time
            logger.info(f"📊 Performance metrics updated - Sharpe: {sharpe_ratio:.2f}, "
                       f"Return: {total_return:.2%}, Max DD: {max_drawdown:.2%}")
        except Exception as e:
            logger.error(f"❌ Performance metrics calculation error: {e}")
    def _calculate_sortino_fallback(self, returns_array: np.ndarray) -> float:
        """Fallback Sortino ratio calculation"""
        try:
            downside_returns = returns_array[returns_array < 0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns)
                return np.mean(returns_array) * 252 / (downside_std * np.sqrt(252)) if downside_std > 0 else 0
            else:
                return float('inf')  # No downside
        except:
            return 0.0
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        if len(self.daily_returns) < 30:
            return 0.0
        try:
            returns_array = np.array(list(self.daily_returns))
            return empyrical.sharpe_ratio(returns_array)
        except:
            returns_array = np.array(list(self.daily_returns))
            if np.std(returns_array) > 0:
                return np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
            return 0.0
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if len(self.portfolio_values) < 2:
            return 0.0
        try:
            values = np.array(list(self.portfolio_values))
            peak = np.maximum.accumulate(values)
            drawdown = (values - peak) / peak
            return abs(np.min(drawdown))
        except:
            return 0.0
    def _calculate_win_rate(self) -> float:
        """Calculate win rate from trades"""
        if not self.trades_history:
            return 0.0
        winning_trades = sum(1 for trade in self.trades_history if trade['pnl'] > 0)
        return winning_trades / len(self.trades_history)
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if not self.trades_history:
            return 0.0
        gross_profit = sum(trade['pnl'] for trade in self.trades_history if trade['pnl'] > 0)
        gross_loss = abs(sum(trade['pnl'] for trade in self.trades_history if trade['pnl'] < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    def _calculate_avg_win_loss(self) -> Tuple[float, float]:
        """Calculate average winning and losing trade"""
        if not self.trades_history:
            return 0.0, 0.0
        winning_trades = [trade['pnl'] for trade in self.trades_history if trade['pnl'] > 0]
        losing_trades = [trade['pnl'] for trade in self.trades_history if trade['pnl'] < 0]
        avg_win = np.mean(winning_trades) if winning_trades else 0.0
        avg_loss = np.mean(losing_trades) if losing_trades else 0.0
        return avg_win, avg_loss
    def _calculate_largest_win_loss(self) -> Tuple[float, float]:
        """Calculate largest winning and losing trade"""
        if not self.trades_history:
            return 0.0, 0.0
        all_pnls = [trade['pnl'] for trade in self.trades_history]
        largest_win = max(all_pnls) if all_pnls else 0.0
        largest_loss = min(all_pnls) if all_pnls else 0.0
        return largest_win, largest_loss
    def _calculate_average_trade_duration(self) -> float:
        """Calculate average trade duration in hours"""
        if not self.trades_history:
            return 0.0
        durations = [trade['duration_hours'] for trade in self.trades_history]
        return np.mean(durations)
    def _calculate_turnover(self) -> float:
        """Calculate portfolio turnover"""
        if not self.trades_history or self.current_capital <= 0:
            return 0.0
        total_trade_value = sum(abs(trade['quantity'] * trade['entry_price']) 
                               for trade in self.trades_history)
        trading_days = len(self.daily_returns) if self.daily_returns else 1
        turnover = (total_trade_value / self.current_capital) * (252 / trading_days)
        return turnover
    def _calculate_market_metrics(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate beta, alpha, and information ratio vs benchmark"""
        if len(self.daily_returns) < 30 or len(self.benchmark_returns) < 30:
            return None, None, None
        try:
            portfolio_returns = np.array(list(self.daily_returns))
            benchmark_returns = np.array(list(self.benchmark_returns))
            min_len = min(len(portfolio_returns), len(benchmark_returns))
            portfolio_returns = portfolio_returns[-min_len:]
            benchmark_returns = benchmark_returns[-min_len:]
            covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else None
            if beta is not None:
                portfolio_annual = np.mean(portfolio_returns) * 252
                benchmark_annual = np.mean(benchmark_returns) * 252
                alpha = portfolio_annual - beta * benchmark_annual
            else:
                alpha = None
            excess_returns = portfolio_returns - benchmark_returns
            tracking_error = np.std(excess_returns) * np.sqrt(252)
            information_ratio = np.mean(excess_returns) * 252 / tracking_error if tracking_error > 0 else None
            return beta, alpha, information_ratio
        except Exception as e:
            logger.error(f"❌ Market metrics calculation error: {e}")
            return None, None, None
    async def _check_risk_alerts(self):
        """Check for risk threshold breaches"""
        try:
            alerts = []
            max_dd = self._calculate_max_drawdown()
            if max_dd > self.risk_thresholds['max_drawdown']:
                alerts.append({
                    'type': 'max_drawdown_breach',
                    'value': max_dd,
                    'threshold': self.risk_thresholds['max_drawdown'],
                    'message': f"Maximum drawdown {max_dd:.2%} exceeds threshold {self.risk_thresholds['max_drawdown']:.2%}",
                    'timestamp': time.time()
                })
            if self.positions:
                max_weight = max(pos.weight for pos in self.positions.values())
                if max_weight > self.risk_thresholds['concentration']:
                    alerts.append({
                        'type': 'concentration_risk',
                        'value': max_weight,
                        'threshold': self.risk_thresholds['concentration'],
                        'message': f"Position concentration {max_weight:.2%} exceeds threshold {self.risk_thresholds['concentration']:.2%}",
                        'timestamp': time.time()
                    })
            if len(self.daily_returns) >= 20:
                returns_array = np.array(list(self.daily_returns))
                var_95 = np.percentile(returns_array, 5)
                if abs(var_95) > self.risk_thresholds['daily_var']:
                    alerts.append({
                        'type': 'var_breach',
                        'value': abs(var_95),
                        'threshold': self.risk_thresholds['daily_var'],
                        'message': f"Daily VaR {abs(var_95):.2%} exceeds threshold {self.risk_thresholds['daily_var']:.2%}",
                        'timestamp': time.time()
                    })
            for alert in alerts:
                self.risk_alerts.append(alert)
                logger.warning(f"⚠️ Risk Alert: {alert['message']}")
                for handler in self.performance_alert_handlers:
                    try:
                        await asyncio.create_task(handler(alert))
                    except Exception as e:
                        logger.error(f"❌ Alert handler error: {e}")
        except Exception as e:
            logger.error(f"❌ Risk alert check error: {e}")
    async def _notify_portfolio_update(self):
        """Notify portfolio update handlers"""
        if self.portfolio_history:
            latest_snapshot = self.portfolio_history[-1]
            for handler in self.portfolio_update_handlers:
                try:
                    await asyncio.create_task(handler(latest_snapshot))
                except Exception as e:
                    logger.error(f"❌ Portfolio update handler error: {e}")
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        try:
            if not self.portfolio_history:
                return {"message": "No portfolio data available"}
            latest = self.portfolio_history[-1]
            position_summary = {}
            for symbol, position in latest.positions.items():
                position_summary[symbol] = {
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'market_value': position.market_value,
                    'unrealized_pnl': position.unrealized_pnl,
                    'weight': position.weight,
                    'duration_hours': position.duration_hours
                }
            recent_returns = list(self.daily_returns)[-30:] if len(self.daily_returns) >= 30 else list(self.daily_returns)
            recent_performance = {
                'avg_daily_return': np.mean(recent_returns) if recent_returns else 0,
                'volatility': np.std(recent_returns) if recent_returns else 0,
                'best_day': max(recent_returns) if recent_returns else 0,
                'worst_day': min(recent_returns) if recent_returns else 0
            }
            return {
                'timestamp': latest.timestamp,
                'total_value': latest.total_value,
                'cash': latest.cash,
                'total_unrealized_pnl': latest.total_unrealized_pnl,
                'total_realized_pnl': latest.total_realized_pnl,
                'cumulative_return': latest.cumulative_return,
                'number_of_positions': latest.number_of_positions,
                'positions': position_summary,
                'performance_metrics': asdict(self.current_metrics) if self.current_metrics else {},
                'recent_performance': recent_performance,
                'risk_alerts': self.risk_alerts[-5:],  # Last 5 alerts
                'trade_summary': {
                    'total_trades': len(self.trades_history),
                    'win_rate': latest.win_rate,
                    'average_trade_duration': latest.average_trade_duration,
                    'recent_trades': self.trades_history[-5:] if self.trades_history else []
                }
            }
        except Exception as e:
            logger.error(f"❌ Portfolio summary error: {e}")
            return {"error": "Unable to generate portfolio summary"}
    def get_performance_report(self, period_days: int = 30) -> Dict:
        """Generate detailed performance report"""
        try:
            if not self.current_metrics:
                return {"message": "Insufficient data for performance report"}
            cutoff_time = time.time() - (period_days * 24 * 3600)
            recent_trades = [trade for trade in self.trades_history if trade['timestamp'] >= cutoff_time]
            if recent_trades:
                winning_trades = [t for t in recent_trades if t['pnl'] > 0]
                losing_trades = [t for t in recent_trades if t['pnl'] < 0]
                trade_analysis = {
                    'total_trades': len(recent_trades),
                    'winning_trades': len(winning_trades),
                    'losing_trades': len(losing_trades),
                    'win_rate': len(winning_trades) / len(recent_trades),
                    'total_pnl': sum(t['pnl'] for t in recent_trades),
                    'best_trade': max(recent_trades, key=lambda x: x['pnl']) if recent_trades else None,
                    'worst_trade': min(recent_trades, key=lambda x: x['pnl']) if recent_trades else None
                }
            else:
                trade_analysis = {'message': 'No trades in selected period'}
            recent_snapshots = [s for s in self.portfolio_history if s.timestamp >= cutoff_time]
            if recent_snapshots:
                value_evolution = {
                    'start_value': recent_snapshots[0].total_value,
                    'end_value': recent_snapshots[-1].total_value,
                    'min_value': min(s.total_value for s in recent_snapshots),
                    'max_value': max(s.total_value for s in recent_snapshots),
                    'period_return': (recent_snapshots[-1].total_value - recent_snapshots[0].total_value) / recent_snapshots[0].total_value
                }
            else:
                value_evolution = {'message': 'No portfolio data in selected period'}
            return {
                'period_days': period_days,
                'metrics': asdict(self.current_metrics),
                'trade_analysis': trade_analysis,
                'value_evolution': value_evolution,
                'risk_alerts_count': len([a for a in self.risk_alerts if a.get('timestamp', 0) >= cutoff_time]),
                'current_positions': len(self.positions),
                'generated_at': time.time()
            }
        except Exception as e:
            logger.error(f"❌ Performance report error: {e}")
            return {"error": "Unable to generate performance report"}
    async def rebalance_portfolio(self, target_weights: Dict[str, float]):
        """Rebalance portfolio to target weights"""
        try:
            logger.info("⚖️ Starting portfolio rebalancing...")
            total_value = sum(pos.market_value for pos in self.positions.values()) + self.cash
            rebalance_orders = []
            for symbol, target_weight in target_weights.items():
                target_value = total_value * target_weight
                if symbol in self.positions:
                    current_value = self.positions[symbol].market_value
                    current_price = self.positions[symbol].current_price
                else:
                    current_value = 0
                    current_price = 50000
                value_difference = target_value - current_value
                quantity_difference = value_difference / current_price
                if abs(quantity_difference) > 0.001:
                    rebalance_orders.append({
                        'symbol': symbol,
                        'quantity': quantity_difference,
                        'target_weight': target_weight,
                        'current_value': current_value,
                        'target_value': target_value
                    })
            logger.info(f"📋 Rebalancing {len(rebalance_orders)} positions")
            return rebalance_orders
        except Exception as e:
            logger.error(f"❌ Rebalancing error: {e}")
            return []
    def save_performance_data(self, filepath: str = None):
        """Save performance data to file"""
        try:
            if not filepath:
                filepath = f"data/performance/portfolio_data_{int(time.time())}.json"
            data = {
                'portfolio_history': [asdict(snapshot) for snapshot in self.portfolio_history],
                'trades_history': self.trades_history,
                'performance_metrics': asdict(self.current_metrics) if self.current_metrics else None,
                'risk_alerts': self.risk_alerts,
                'daily_returns': list(self.daily_returns),
                'portfolio_values': list(self.portfolio_values),
                'initial_capital': self.initial_capital,
                'current_capital': self.current_capital
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"💾 Performance data saved to {filepath}")
        except Exception as e:
            logger.error(f"❌ Save performance data error: {e}")
    def load_performance_data(self, filepath: str):
        """Load performance data from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.trades_history = data.get('trades_history', [])
            self.risk_alerts = data.get('risk_alerts', [])
            self.initial_capital = data.get('initial_capital', self.initial_capital)
            self.current_capital = data.get('current_capital', self.initial_capital)
            if 'daily_returns' in data:
                self.daily_returns = deque(data['daily_returns'], maxlen=self.performance_window)
            if 'portfolio_values' in data:
                self.portfolio_values = deque(data['portfolio_values'], maxlen=self.performance_window)
            logger.info(f"📂 Performance data loaded from {filepath}")
        except Exception as e:
            logger.error(f"❌ Load performance data error: {e}")