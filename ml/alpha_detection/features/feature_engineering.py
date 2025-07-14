import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats
import talib
class FeatureEngineer:
    """
    Advanced feature engineering for alpha detection
    """
    def __init__(self):
        self.feature_cache = {}
    def engineer_price_features(self, 
                              prices: List[float], 
                              volumes: List[float],
                              timestamps: List[float]) -> Dict[str, float]:
        """
        Engineer comprehensive price-based features
        """
        if len(prices) < 20:
            return {}
        prices_array = np.array(prices)
        volumes_array = np.array(volumes) if volumes else np.ones(len(prices))
        features = {}
        features.update(self._calculate_momentum_features(prices_array))
        features.update(self._calculate_volatility_features(prices_array))
        features.update(self._calculate_technical_features(prices_array))
        features.update(self._calculate_volume_features(prices_array, volumes_array))
        features.update(self._calculate_statistical_features(prices_array))
        return features
    def _calculate_momentum_features(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate momentum-based features"""
        features = {}
        for period in [1, 3, 5, 10, 20]:
            if len(prices) > period:
                ret = (prices[-1] - prices[-period-1]) / prices[-period-1]
                features[f'return_{period}'] = ret
        if len(prices) > 10:
            short_momentum = (prices[-1] - prices[-6]) / prices[-6]
            long_momentum = (prices[-1] - prices[-11]) / prices[-11]
            features['momentum_acceleration'] = short_momentum - long_momentum
        if len(prices) > 20:
            x = np.arange(len(prices[-20:]))
            slope, _, r_value, _, _ = stats.linregress(x, prices[-20:])
            features['trend_slope'] = slope
            features['trend_strength'] = r_value ** 2
        return features
    def _calculate_volatility_features(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate volatility-based features"""
        features = {}
        for window in [5, 10, 20]:
            if len(prices) > window:
                returns = np.diff(prices[-window-1:]) / prices[-window-1:-1]
                features[f'volatility_{window}'] = np.std(returns)
        if len(prices) > 40:
            vol_series = []
            for i in range(20, len(prices), 5):
                if i + 5 <= len(prices):
                    window_returns = np.diff(prices[i-5:i+1]) / prices[i-5:i]
                    vol_series.append(np.std(window_returns))
            if len(vol_series) > 2:
                features['vol_of_vol'] = np.std(vol_series)
        if len(prices) > 20:
            returns = np.diff(prices) / prices[:-1]
            features['skewness'] = stats.skew(returns[-20:])
            features['kurtosis'] = stats.kurtosis(returns[-20:])
        return features
    def _calculate_technical_features(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate technical indicator features"""
        features = {}
        if len(prices) < 20:
            return features
        try:
            if len(prices) >= 14:
                rsi = talib.RSI(prices.astype(float), timeperiod=14)
                if not np.isnan(rsi[-1]):
                    features['rsi'] = rsi[-1]
            if len(prices) >= 26:
                macd, macd_signal, macd_hist = talib.MACD(prices.astype(float))
                if not np.isnan(macd[-1]):
                    features['macd'] = macd[-1]
                    features['macd_signal'] = macd_signal[-1]
                    features['macd_histogram'] = macd_hist[-1]
            if len(prices) >= 20:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(prices.astype(float))
                if not np.isnan(bb_upper[-1]):
                    bb_position = (prices[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
                    features['bb_position'] = bb_position
                    features['bb_width'] = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]
            if len(prices) >= 14:
                high = low = close = prices.astype(float)
                slowk, slowd = talib.STOCH(high, low, close)
                if not np.isnan(slowk[-1]):
                    features['stoch_k'] = slowk[-1]
                    features['stoch_d'] = slowd[-1]
        except Exception as e:
            features.update(self._calculate_simple_technical_features(prices))
        return features
    def _calculate_simple_technical_features(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate simple technical features without TA-Lib"""
        features = {}
        if len(prices) >= 14:
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                features['rsi'] = rsi
        for period in [5, 10, 20]:
            if len(prices) >= period:
                ma = np.mean(prices[-period:])
                features[f'sma_{period}_ratio'] = prices[-1] / ma
        return features
    def _calculate_volume_features(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, float]:
        """Calculate volume-based features"""
        features = {}
        if len(volumes) < 5:
            return features
        features['volume_sma_ratio'] = volumes[-1] / np.mean(volumes[-5:]) if np.mean(volumes[-5:]) > 0 else 1
        if len(prices) >= 10 and len(volumes) >= 10:
            price_changes = np.diff(prices[-10:])
            volume_changes = volumes[-9:]
            if len(price_changes) == len(volume_changes):
                corr = np.corrcoef(price_changes, volume_changes)[0, 1]
                features['price_volume_corr'] = corr if not np.isnan(corr) else 0
        if len(prices) >= 10:
            obv = 0
            for i in range(1, min(10, len(prices))):
                if prices[-i] > prices[-i-1]:
                    obv += volumes[-i]
                elif prices[-i] < prices[-i-1]:
                    obv -= volumes[-i]
            features['obv_momentum'] = obv
        return features
    def _calculate_statistical_features(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate statistical features"""
        features = {}
        if len(prices) < 10:
            return features
        features['price_percentile_20'] = np.percentile(prices[-20:], 80) if len(prices) >= 20 else 0
        features['price_percentile_50'] = np.percentile(prices[-20:], 50) if len(prices) >= 20 else 0
        if len(prices) >= 20:
            mean_price = np.mean(prices[-20:])
            std_price = np.std(prices[-20:])
            if std_price > 0:
                features['price_zscore'] = (prices[-1] - mean_price) / std_price
        if len(prices) >= 50:
            features['hurst_exponent'] = self._calculate_hurst_exponent(prices[-50:])
        return features
    def _calculate_hurst_exponent(self, prices: np.ndarray) -> float:
        """Calculate simplified Hurst exponent"""
        try:
            log_returns = np.diff(np.log(prices))
            lags = [5, 10, 20]
            rs_values = []
            for lag in lags:
                if len(log_returns) >= lag:
                    n_blocks = len(log_returns) // lag
                    rs_block = []
                    for i in range(n_blocks):
                        block = log_returns[i*lag:(i+1)*lag]
                        mean_block = np.mean(block)
                        cum_dev = np.cumsum(block - mean_block)
                        R = np.max(cum_dev) - np.min(cum_dev)
                        S = np.std(block)
                        if S > 0:
                            rs_block.append(R / S)
                    if rs_block:
                        rs_values.append(np.mean(rs_block))
            if len(rs_values) >= 2:
                log_lags = np.log(lags[:len(rs_values)])
                log_rs = np.log(rs_values)
                slope, _, _, _, _ = stats.linregress(log_lags, log_rs)
                return slope
        except Exception:
            pass
        return 0.5