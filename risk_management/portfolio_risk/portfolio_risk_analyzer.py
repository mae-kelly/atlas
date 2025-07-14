import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats
from scipy.linalg import sqrtm
import cvxpy as cp
from dataclasses import dataclass
@dataclass
class PortfolioRiskMetrics:
    portfolio_var: float
    portfolio_cvar: float
    portfolio_volatility: float
    maximum_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    tail_ratio: float
    concentration_risk: float
    liquidity_risk: float
class PortfolioRiskAnalyzer:
    """
    Comprehensive portfolio risk analysis
    """
    def __init__(self, confidence_level: float = 0.05):
        self.confidence_level = confidence_level
    def calculate_portfolio_var(self,
                              returns: np.ndarray,
                              weights: np.ndarray,
                              method: str = 'historical') -> float:
        """
        Calculate Portfolio Value at Risk
        Methods: 'historical', 'parametric', 'cornish_fisher'
        """
        portfolio_returns = np.dot(returns, weights)
        if method == 'historical':
            return np.percentile(portfolio_returns, self.confidence_level * 100)
        elif method == 'parametric':
            mean_return = np.mean(portfolio_returns)
            std_return = np.std(portfolio_returns)
            return mean_return + std_return * stats.norm.ppf(self.confidence_level)
        elif method == 'cornish_fisher':
            return self._cornish_fisher_var(portfolio_returns)
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    def _cornish_fisher_var(self, returns: np.ndarray) -> float:
        """
        Cornish-Fisher expansion for VaR (accounts for skewness and kurtosis)
        """
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        skewness = stats.skew(returns)
        kurt = stats.kurtosis(returns)
        z = stats.norm.ppf(self.confidence_level)
        z_cf = (z + (z**2 - 1) * skewness / 6 + 
                (z**3 - 3*z) * kurt / 24 - 
                (2*z**3 - 5*z) * skewness**2 / 36)
        return mean_ret + std_ret * z_cf
    def calculate_portfolio_cvar(self,
                               returns: np.ndarray,
                               weights: np.ndarray) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)
        """
        portfolio_returns = np.dot(returns, weights)
        var = self.calculate_portfolio_var(returns, weights, method='historical')
        tail_returns = portfolio_returns[portfolio_returns <= var]
        if len(tail_returns) > 0:
            return np.mean(tail_returns)
        else:
            return var
    def calculate_maximum_drawdown(self, portfolio_values: np.ndarray) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown and its duration
        Returns: (max_drawdown, start_index, end_index)
        """
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_dd = np.min(drawdown)
        end_idx = np.argmin(drawdown)
        start_idx = np.argmax(portfolio_values[:end_idx])
        return abs(max_dd), start_idx, end_idx
    def calculate_risk_adjusted_returns(self,
                                      returns: np.ndarray,
                                      risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Calculate various risk-adjusted return metrics
        """
        annual_return = np.mean(returns) * 252
        annual_volatility = np.std(returns) * np.sqrt(252)
        downside_returns = returns[returns < 0]
        downside_volatility = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        sortino = (annual_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        portfolio_values = np.cumprod(1 + returns)
        max_dd, _, _ = self.calculate_maximum_drawdown(portfolio_values)
        calmar = annual_return / max_dd if max_dd > 0 else 0
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_dd
        }
    def calculate_concentration_risk(self, weights: np.ndarray) -> float:
        """
        Calculate portfolio concentration using Herfindahl-Hirschman Index
        """
        normalized_weights = np.abs(weights) / np.sum(np.abs(weights))
        hhi = np.sum(normalized_weights ** 2)
        n = len(weights)
        min_hhi = 1 / n
        concentration_risk = (hhi - min_hhi) / (1 - min_hhi)
        return concentration_risk
    def calculate_tail_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate tail ratio (95th percentile / 5th percentile)
        """
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        if p5 != 0:
            return abs(p95 / p5)
        else:
            return np.inf
    def calculate_portfolio_beta(self,
                               portfolio_returns: np.ndarray,
                               market_returns: np.ndarray) -> float:
        """
        Calculate portfolio beta against market
        """
        if len(portfolio_returns) != len(market_returns):
            min_len = min(len(portfolio_returns), len(market_returns))
            portfolio_returns = portfolio_returns[:min_len]
            market_returns = market_returns[:min_len]
        covariance = np.cov(portfolio_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        if market_variance > 0:
            return covariance / market_variance
        else:
            return 0.0
    def optimize_risk_parity(self,
                           cov_matrix: np.ndarray,
                           target_risk: np.ndarray = None) -> np.ndarray:
        """
        Optimize portfolio for risk parity
        """
        n = cov_matrix.shape[0]
        if target_risk is None:
            target_risk = np.ones(n) / n
        w = cp.Variable(n)
        portfolio_variance = cp.quad_form(w, cov_matrix)
        marginal_contrib = cov_matrix @ w
        risk_contrib = cp.multiply(w, marginal_contrib) / portfolio_variance
        objective = cp.Minimize(cp.sum_squares(risk_contrib - target_risk))
        constraints = [
            cp.sum(w) == 1,
            w >= 0.001,
            w <= 0.3
        ]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        if problem.status == cp.OPTIMAL:
            return w.value
        else:
            return np.ones(n) / n
    def stress_test_portfolio(self,
                            returns: np.ndarray,
                            weights: np.ndarray,
                            scenarios: List[Dict]) -> Dict:
        """
        Stress test portfolio under various scenarios
        """
        results = {}
        for i, scenario in enumerate(scenarios):
            scenario_name = scenario.get('name', f'Scenario_{i+1}')
            if 'shock_magnitude' in scenario:
                shocked_returns = returns + scenario['shock_magnitude']
                portfolio_return = np.dot(shocked_returns.mean(axis=0), weights)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(np.cov(shocked_returns.T), weights)))
            elif 'correlation_shock' in scenario:
                corr_matrix = np.corrcoef(returns.T)
                shocked_corr = corr_matrix * scenario['correlation_shock']
                np.fill_diagonal(shocked_corr, 1.0)
                stds = np.std(returns, axis=0)
                shocked_cov = np.outer(stds, stds) * shocked_corr
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(shocked_cov, weights)))
                portfolio_return = np.dot(returns.mean(axis=0), weights)
            else:
                portfolio_return = np.dot(returns.mean(axis=0), weights)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(np.cov(returns.T), weights)))
            results[scenario_name] = {
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
            }
        return results
    def calculate_comprehensive_risk_metrics(self,
                                           returns: np.ndarray,
                                           weights: np.ndarray,
                                           portfolio_values: np.ndarray = None) -> PortfolioRiskMetrics:
        """
        Calculate comprehensive portfolio risk metrics
        """
        var = self.calculate_portfolio_var(returns, weights)
        cvar = self.calculate_portfolio_cvar(returns, weights)
        portfolio_returns = np.dot(returns, weights)
        volatility = np.std(portfolio_returns) * np.sqrt(252)
        risk_adjusted = self.calculate_risk_adjusted_returns(portfolio_returns)
        if portfolio_values is not None:
            max_dd, _, _ = self.calculate_maximum_drawdown(portfolio_values)
        else:
            portfolio_values = np.cumprod(1 + portfolio_returns)
            max_dd, _, _ = self.calculate_maximum_drawdown(portfolio_values)
        concentration = self.calculate_concentration_risk(weights)
        tail_ratio = self.calculate_tail_ratio(portfolio_returns)
        return PortfolioRiskMetrics(
            portfolio_var=var,
            portfolio_cvar=cvar,
            portfolio_volatility=volatility,
            maximum_drawdown=max_dd,
            sharpe_ratio=risk_adjusted['sharpe_ratio'],
            sortino_ratio=risk_adjusted['sortino_ratio'],
            calmar_ratio=risk_adjusted['calmar_ratio'],
            tail_ratio=tail_ratio,
            concentration_risk=concentration,
            liquidity_risk=0.0
        )