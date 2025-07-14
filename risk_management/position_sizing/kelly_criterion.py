import numpy as np
from typing import Tuple, Optional
from scipy.optimize import minimize_scalar
import math

class KellyCriterion:
    """
    Advanced Kelly Criterion implementation for position sizing
    """
    
    def __init__(self):
        self.min_bet_size = 0.001  # 0.1% minimum
        self.max_bet_size = 0.25   # 25% maximum (fractional Kelly)
    
    def calculate_kelly_fraction(self, 
                               win_probability: float,
                               win_amount: float,
                               loss_amount: float) -> float:
        """
        Classic Kelly formula: f* = (bp - q) / b
        where:
        - b = win_amount / loss_amount (odds)
        - p = win_probability
        - q = 1 - p (loss probability)
        """
        if win_probability <= 0 or win_probability >= 1:
            return 0.0
        
        if win_amount <= 0 or loss_amount <= 0:
            return 0.0
        
        b = win_amount / loss_amount  # Odds
        p = win_probability
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Apply safety constraints
        kelly_fraction = max(0, min(self.max_bet_size, kelly_fraction))
        
        return kelly_fraction
    
    def calculate_continuous_kelly(self,
                                 expected_return: float,
                                 variance: float) -> float:
        """
        Continuous Kelly formula for normally distributed returns
        f* = μ / σ²
        """
        if variance <= 0:
            return 0.0
        
        kelly_fraction = expected_return / variance
        
        # Apply safety constraints
        kelly_fraction = max(0, min(self.max_bet_size, kelly_fraction))
        
        return kelly_fraction
    
    def calculate_fractional_kelly(self,
                                 kelly_fraction: float,
                                 fraction: float = 0.25) -> float:
        """
        Apply fractional Kelly for risk management
        Common fractions: 0.25 (quarter Kelly), 0.5 (half Kelly)
        """
        return kelly_fraction * fraction
    
    def calculate_kelly_with_drawdown_constraint(self,
                                               win_prob: float,
                                               win_amount: float,
                                               loss_amount: float,
                                               max_drawdown: float = 0.2) -> float:
        """
        Calculate Kelly fraction with maximum drawdown constraint
        """
        base_kelly = self.calculate_kelly_fraction(win_prob, win_amount, loss_amount)
        
        # Estimate probability of hitting max drawdown
        if base_kelly > 0:
            # Approximate formula for drawdown probability
            dd_prob = self._estimate_drawdown_probability(base_kelly, win_prob, max_drawdown)
            
            # Reduce Kelly fraction if drawdown probability is too high
            if dd_prob > 0.05:  # 5% max probability of hitting drawdown
                reduction_factor = 0.05 / dd_prob
                base_kelly *= reduction_factor
        
        return max(self.min_bet_size, min(self.max_bet_size, base_kelly))
    
    def _estimate_drawdown_probability(self,
                                     kelly_fraction: float,
                                     win_prob: float,
                                     max_drawdown: float) -> float:
        """
        Estimate probability of hitting maximum drawdown
        Using simplified approximation
        """
        if kelly_fraction <= 0:
            return 0.0
        
        # Simplified formula based on gambler's ruin problem
        # This is an approximation - exact calculation requires simulation
        
        try:
            q_over_p = (1 - win_prob) / win_prob
            
            if q_over_p == 1:  # Equal probability
                prob = 1.0 / (1.0 + 1.0 / max_drawdown)
            else:
                ratio = q_over_p ** (max_drawdown / kelly_fraction)
                prob = ratio / (1 + ratio)
            
            return min(1.0, max(0.0, prob))
            
        except (ZeroDivisionError, OverflowError):
            return 1.0  # Conservative estimate
    
    def optimize_kelly_with_utility(self,
                                  win_prob: float,
                                  win_amount: float,
                                  loss_amount: float,
                                  risk_aversion: float = 2.0) -> float:
        """
        Optimize Kelly fraction considering utility function
        Uses CRRA utility: U(W) = W^(1-γ) / (1-γ)
        where γ is risk aversion parameter
        """
        def negative_expected_utility(f):
            if f <= 0 or f >= 1:
                return float('inf')
            
            # Expected utility calculation
            win_wealth = 1 + f * win_amount
            loss_wealth = 1 - f * loss_amount
            
            if win_wealth <= 0 or loss_wealth <= 0:
                return float('inf')
            
            if risk_aversion == 1:  # Log utility
                expected_utility = (
                    win_prob * math.log(win_wealth) +
                    (1 - win_prob) * math.log(loss_wealth)
                )
            else:  # CRRA utility
                expected_utility = (
                    win_prob * (win_wealth ** (1 - risk_aversion)) / (1 - risk_aversion) +
                    (1 - win_prob) * (loss_wealth ** (1 - risk_aversion)) / (1 - risk_aversion)
                )
            
            return -expected_utility
        
        # Optimize between min and max bet sizes
        result = minimize_scalar(
            negative_expected_utility,
            bounds=(self.min_bet_size, self.max_bet_size),
            method='bounded'
        )
        
        if result.success:
            return result.x
        else:
            # Fallback to regular Kelly
            return self.calculate_kelly_fraction(win_prob, win_amount, loss_amount)
    
    def simulate_kelly_performance(self,
                                 kelly_fraction: float,
                                 win_prob: float,
                                 win_amount: float,
                                 loss_amount: float,
                                 num_trades: int = 1000,
                                 num_simulations: int = 1000) -> dict:
        """
        Monte Carlo simulation of Kelly strategy performance
        """
        final_wealths = []
        max_drawdowns = []
        
        for _ in range(num_simulations):
            wealth = 1.0
            peak_wealth = 1.0
            max_dd = 0.0
            
            for _ in range(num_trades):
                if np.random.random() < win_prob:
                    # Win
                    wealth *= (1 + kelly_fraction * win_amount)
                else:
                    # Loss
                    wealth *= (1 - kelly_fraction * loss_amount)
                
                # Track drawdown
                if wealth > peak_wealth:
                    peak_wealth = wealth
                
                current_dd = (peak_wealth - wealth) / peak_wealth
                max_dd = max(max_dd, current_dd)
            
            final_wealths.append(wealth)
            max_drawdowns.append(max_dd)
        
        return {
            'final_wealth_mean': np.mean(final_wealths),
            'final_wealth_std': np.std(final_wealths),
            'final_wealth_median': np.median(final_wealths),
            'max_drawdown_mean': np.mean(max_drawdowns),
            'max_drawdown_95th': np.percentile(max_drawdowns, 95),
            'probability_of_ruin': sum(1 for w in final_wealths if w < 0.1) / len(final_wealths),
            'probability_of_doubling': sum(1 for w in final_wealths if w > 2.0) / len(final_wealths)
        }
