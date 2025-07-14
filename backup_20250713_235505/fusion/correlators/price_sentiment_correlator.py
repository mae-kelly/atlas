import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Tuple
from dataclasses import dataclass
@dataclass
class CorrelationMetrics:
    pearson_correlation: float
    spearman_correlation: float
    kendall_tau: float
    p_value: float
    confidence_interval: Tuple[float, float]
    sample_size: int
class PriceSentimentCorrelator:
    """
    Advanced correlation analysis between price movements and sentiment
    """
    def __init__(self, min_samples: int = 5):
        self.min_samples = min_samples
    def calculate_advanced_correlation(self, 
                                     price_series: List[float], 
                                     sentiment_series: List[float]) -> CorrelationMetrics:
        """
        Calculate multiple correlation metrics with statistical significance
        """
        if len(price_series) < self.min_samples or len(sentiment_series) < self.min_samples:
            return self._empty_correlation()
        min_len = min(len(price_series), len(sentiment_series))
        prices = np.array(price_series[:min_len])
        sentiments = np.array(sentiment_series[:min_len])
        mask = ~(np.isnan(prices) | np.isnan(sentiments))
        prices = prices[mask]
        sentiments = sentiments[mask]
        if len(prices) < self.min_samples:
            return self._empty_correlation()
        try:
            pearson_r, pearson_p = stats.pearsonr(prices, sentiments)
            spearman_r, spearman_p = stats.spearmanr(prices, sentiments)
            kendall_tau, kendall_p = stats.kendalltau(prices, sentiments)
            n = len(prices)
            r = pearson_r
            fisher_z = np.arctanh(r)
            se = 1 / np.sqrt(n - 3)
            ci_low = np.tanh(fisher_z - 1.96 * se)
            ci_high = np.tanh(fisher_z + 1.96 * se)
            return CorrelationMetrics(
                pearson_correlation=pearson_r,
                spearman_correlation=spearman_r,
                kendall_tau=kendall_tau,
                p_value=min(pearson_p, spearman_p, kendall_p),
                confidence_interval=(ci_low, ci_high),
                sample_size=n
            )
        except Exception as e:
            return self._empty_correlation()
    def _empty_correlation(self) -> CorrelationMetrics:
        """Return empty correlation metrics"""
        return CorrelationMetrics(
            pearson_correlation=0.0,
            spearman_correlation=0.0,
            kendall_tau=0.0,
            p_value=1.0,
            confidence_interval=(0.0, 0.0),
            sample_size=0
        )
    def rolling_correlation(self, 
                          price_series: List[float], 
                          sentiment_series: List[float],
                          window: int = 10) -> List[float]:
        """
        Calculate rolling correlation over a sliding window
        """
        correlations = []
        for i in range(window, min(len(price_series), len(sentiment_series)) + 1):
            price_window = price_series[i-window:i]
            sentiment_window = sentiment_series[i-window:i]
            if len(price_window) >= self.min_samples:
                try:
                    corr, _ = stats.pearsonr(price_window, sentiment_window)
                    correlations.append(corr if not np.isnan(corr) else 0.0)
                except:
                    correlations.append(0.0)
            else:
                correlations.append(0.0)
        return correlations