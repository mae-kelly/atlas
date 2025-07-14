import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import empyrical
from scipy import stats
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
class PerformanceAnalytics:
    """
    Advanced portfolio performance analytics and visualization
    """
    def __init__(self):
        self.risk_free_rate = 0.02
    def calculate_risk_metrics(self, returns: List[float], benchmark_returns: List[float] = None) -> Dict:
        """
        Calculate comprehensive risk metrics
        """
        returns_array = np.array(returns)
        if len(returns_array) < 30:
            return {"error": "Insufficient data for risk analysis"}
        metrics = {}
        try:
            metrics['total_return'] = empyrical.cum_returns_final(returns_array)
            metrics['annual_return'] = empyrical.annual_return(returns_array)
            metrics['annual_volatility'] = empyrical.annual_volatility(returns_array)
            metrics['sharpe_ratio'] = empyrical.sharpe_ratio(returns_array, risk_free=self.risk_free_rate/252)
            metrics['sortino_ratio'] = empyrical.sortino_ratio(returns_array, required_return=0)
            metrics['calmar_ratio'] = empyrical.calmar_ratio(returns_array)
            metrics['max_drawdown'] = empyrical.max_drawdown(returns_array)
        except Exception as e:
            metrics['annual_return'] = np.mean(returns_array) * 252
            metrics['annual_volatility'] = np.std(returns_array) * np.sqrt(252)
            metrics['sharpe_ratio'] = metrics['annual_return'] / metrics['annual_volatility'] if metrics['annual_volatility'] > 0 else 0
            metrics['max_drawdown'] = self._calculate_max_drawdown_fallback(returns_array)
        metrics['var_95'] = np.percentile(returns_array, 5)
        metrics['cvar_95'] = returns_array[returns_array <= metrics['var_95']].mean()
        metrics['skewness'] = stats.skew(returns_array)
        metrics['kurtosis'] = stats.kurtosis(returns_array)
        if benchmark_returns is not None and len(benchmark_returns) == len(returns_array):
            benchmark_array = np.array(benchmark_returns)
            metrics['beta'] = self._calculate_beta(returns_array, benchmark_array)
            metrics['alpha'] = self._calculate_alpha(returns_array, benchmark_array, metrics['beta'])
            metrics['information_ratio'] = self._calculate_information_ratio(returns_array, benchmark_array)
            metrics['tracking_error'] = np.std(returns_array - benchmark_array) * np.sqrt(252)
            up_capture, down_capture = self._calculate_capture_ratios(returns_array, benchmark_array)
            metrics['up_capture'] = up_capture
            metrics['down_capture'] = down_capture
        return metrics
    def _calculate_max_drawdown_fallback(self, returns_array: np.ndarray) -> float:
        """Fallback max drawdown calculation"""
        try:
            cumulative = np.cumprod(1 + returns_array)
            peak = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - peak) / peak
            return abs(np.min(drawdown))
        except:
            return 0.0
    def _calculate_beta(self, returns: np.ndarray, benchmark: np.ndarray) -> float:
        """Calculate portfolio beta"""
        try:
            covariance = np.cov(returns, benchmark)[0, 1]
            benchmark_variance = np.var(benchmark)
            return covariance / benchmark_variance if benchmark_variance > 0 else 0
        except:
            return 0.0
    def _calculate_alpha(self, returns: np.ndarray, benchmark: np.ndarray, beta: float) -> float:
        """Calculate portfolio alpha"""
        try:
            portfolio_return = np.mean(returns) * 252
            benchmark_return = np.mean(benchmark) * 252
            return portfolio_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
        except:
            return 0.0
    def _calculate_information_ratio(self, returns: np.ndarray, benchmark: np.ndarray) -> float:
        """Calculate information ratio"""
        try:
            excess_returns = returns - benchmark
            tracking_error = np.std(excess_returns) * np.sqrt(252)
            return np.mean(excess_returns) * 252 / tracking_error if tracking_error > 0 else 0
        except:
            return 0.0
    def _calculate_capture_ratios(self, returns: np.ndarray, benchmark: np.ndarray) -> Tuple[float, float]:
        """Calculate up/down capture ratios"""
        try:
            up_market = benchmark > 0
            down_market = benchmark < 0
            if np.sum(up_market) > 0:
                up_capture = np.mean(returns[up_market]) / np.mean(benchmark[up_market])
            else:
                up_capture = 0
            if np.sum(down_market) > 0:
                down_capture = np.mean(returns[down_market]) / np.mean(benchmark[down_market])
            else:
                down_capture = 0
            return up_capture, down_capture
        except:
            return 0.0, 0.0
    def analyze_trade_performance(self, trades: List[Dict]) -> Dict:
        """
        Analyze individual trade performance
        """
        if not trades:
            return {"error": "No trades to analyze"}
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        analysis = {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades) if trades else 0
        }
        all_pnl = [t['pnl'] for t in trades]
        analysis['total_pnl'] = sum(all_pnl)
        analysis['average_pnl'] = np.mean(all_pnl)
        analysis['median_pnl'] = np.median(all_pnl)
        analysis['pnl_std'] = np.std(all_pnl)
        if winning_trades:
            win_pnl = [t['pnl'] for t in winning_trades]
            analysis['average_win'] = np.mean(win_pnl)
            analysis['largest_win'] = max(win_pnl)
            analysis['win_std'] = np.std(win_pnl)
        if losing_trades:
            loss_pnl = [t['pnl'] for t in losing_trades]
            analysis['average_loss'] = np.mean(loss_pnl)
            analysis['largest_loss'] = min(loss_pnl)
            analysis['loss_std'] = np.std(loss_pnl)
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        analysis['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        durations = [t['duration_hours'] for t in trades if 'duration_hours' in t]
        if durations:
            analysis['average_duration_hours'] = np.mean(durations)
            analysis['median_duration_hours'] = np.median(durations)
            analysis['min_duration_hours'] = min(durations)
            analysis['max_duration_hours'] = max(durations)
        analysis['max_consecutive_wins'] = self._calculate_max_consecutive(trades, 'win')
        analysis['max_consecutive_losses'] = self._calculate_max_consecutive(trades, 'loss')
        if any('strategy_id' in t for t in trades):
            strategy_performance = {}
            for trade in trades:
                strategy = trade.get('strategy_id', 'unknown')
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = {'trades': 0, 'pnl': 0, 'wins': 0}
                strategy_performance[strategy]['trades'] += 1
                strategy_performance[strategy]['pnl'] += trade['pnl']
                if trade['pnl'] > 0:
                    strategy_performance[strategy]['wins'] += 1
            for strategy, stats in strategy_performance.items():
                stats['win_rate'] = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
            analysis['strategy_breakdown'] = strategy_performance
        return analysis
    def _calculate_max_consecutive(self, trades: List[Dict], trade_type: str) -> int:
        """Calculate maximum consecutive wins or losses"""
        if not trades:
            return 0
        max_consecutive = 0
        current_consecutive = 0
        for trade in sorted(trades, key=lambda x: x.get('timestamp', 0)):
            is_target_type = (trade_type == 'win' and trade['pnl'] > 0) or (trade_type == 'loss' and trade['pnl'] < 0)
            if is_target_type:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        return max_consecutive
    def create_performance_dashboard(self, 
                                   portfolio_values: List[float], 
                                   returns: List[float],
                                   positions: Dict,
                                   trades: List[Dict]):
        """
        Create comprehensive performance dashboard
        """
        if not PLOTLY_AVAILABLE:
            return {"error": "Plotly not available for dashboard creation"}
        try:
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Portfolio Value', 'Daily Returns', 
                              'Position Allocation', 'Drawdown',
                              'Trade P&L Distribution', 'Rolling Sharpe Ratio'),
                specs=[[{"secondary_y": True}, {"secondary_y": False}],
                       [{"type": "pie"}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            fig.add_trace(
                go.Scatter(y=portfolio_values, name="Portfolio Value", line=dict(color="blue")),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(y=returns, name="Daily Returns", marker_color="green"),
                row=1, col=2
            )
            if positions:
                symbols = list(positions.keys())
                weights = [pos.weight for pos in positions.values()]
                fig.add_trace(
                    go.Pie(labels=symbols, values=weights, name="Allocation"),
                    row=2, col=1
                )
            if portfolio_values:
                values = np.array(portfolio_values)
                peak = np.maximum.accumulate(values)
                drawdown = (values - peak) / peak
                fig.add_trace(
                    go.Scatter(y=drawdown, name="Drawdown", fill='tonexty', 
                              fillcolor="rgba(255,0,0,0.3)", line=dict(color="red")),
                    row=2, col=2
                )
            if trades:
                pnl_values = [t['pnl'] for t in trades]
                fig.add_trace(
                    go.Histogram(x=pnl_values, name="Trade P&L", nbinsx=30),
                    row=3, col=1
                )
            if len(returns) >= 30:
                rolling_sharpe = []
                for i in range(30, len(returns)):
                    window_returns = returns[i-30:i]
                    sharpe = np.mean(window_returns) / np.std(window_returns) * np.sqrt(252) if np.std(window_returns) > 0 else 0
                    rolling_sharpe.append(sharpe)
                fig.add_trace(
                    go.Scatter(y=rolling_sharpe, name="30-Day Rolling Sharpe", 
                              line=dict(color="purple")),
                    row=3, col=2
                )
            fig.update_layout(
                height=1000,
                title_text="Portfolio Performance Dashboard",
                showlegend=True
            )
            return fig
        except Exception as e:
            return {"error": f"Dashboard creation failed: {e}"}
    def generate_risk_report(self, returns: List[float], positions: Dict) -> Dict:
        """
        Generate comprehensive risk assessment report
        """
        if len(returns) < 30:
            return {"error": "Insufficient data for risk analysis"}
        returns_array = np.array(returns)
        var_levels = [0.01, 0.05, 0.10]
        var_analysis = {}
        for level in var_levels:
            var_analysis[f'var_{int(level*100)}'] = np.percentile(returns_array, level * 100)
            var_threshold = var_analysis[f'var_{int(level*100)}']
            cvar = returns_array[returns_array <= var_threshold].mean()
            var_analysis[f'cvar_{int(level*100)}'] = cvar
        stress_scenarios = {
            '2008_crisis': -0.15,      # -15% shock
            'covid_crash': -0.12,      # -12% shock  
            'flash_crash': -0.08,      # -8% shock
            'black_monday': -0.20      # -20% shock
        }
        stress_results = {}
        portfolio_value = 100000
        for scenario, shock in stress_scenarios.items():
            stressed_value = portfolio_value * (1 + shock)
            stress_results[scenario] = {
                'shock_percentage': shock,
                'stressed_value': stressed_value,
                'loss_amount': portfolio_value - stressed_value
            }
        concentration_risk = self._analyze_concentration_risk(positions)
        correlation_risk = self._estimate_correlation_risk(positions)
        return {
            'var_analysis': var_analysis,
            'stress_test_results': stress_results,
            'concentration_risk': concentration_risk,
            'correlation_risk': correlation_risk,
            'risk_metrics': self.calculate_risk_metrics(returns),
            'generated_at': pd.Timestamp.now().isoformat()
        }
    def _analyze_concentration_risk(self, positions: Dict) -> Dict:
        """Analyze portfolio concentration risk"""
        if not positions:
            return {"message": "No positions to analyze"}
        weights = [pos.weight for pos in positions.values()]
        hhi = sum(w**2 for w in weights)
        effective_positions = 1 / hhi if hhi > 0 else 0
        max_weight = max(weights) if weights else 0
        top_3_weight = sum(sorted(weights, reverse=True)[:3])
        return {
            'hhi_index': hhi,
            'effective_positions': effective_positions,
            'max_position_weight': max_weight,
            'top_3_concentration': top_3_weight,
            'number_of_positions': len(positions),
            'concentration_level': self._classify_concentration(hhi)
        }
    def _classify_concentration(self, hhi: float) -> str:
        """Classify portfolio concentration level"""
        if hhi < 0.15:
            return "Low"
        elif hhi < 0.25:
            return "Moderate"
        elif hhi < 0.50:
            return "High"
        else:
            return "Very High"
    def _estimate_correlation_risk(self, positions: Dict) -> Dict:
        """Estimate correlation risk (simplified)"""
        if len(positions) < 2:
            return {"message": "Insufficient positions for correlation analysis"}
        crypto_positions = sum(1 for symbol in positions.keys() if 'BTC' in symbol or 'ETH' in symbol or 'USDT' in symbol)
        total_positions = len(positions)
        crypto_concentration = crypto_positions / total_positions if total_positions > 0 else 0
        if crypto_concentration > 0.8:
            risk_level = "High"
        elif crypto_concentration > 0.5:
            risk_level = "Moderate" 
        else:
            risk_level = "Low"
        return {
            'estimated_avg_correlation': 0.7 if crypto_concentration > 0.5 else 0.3,
            'crypto_concentration': crypto_concentration,
            'correlation_risk_level': risk_level,
            'diversification_score': 1 - crypto_concentration
        }