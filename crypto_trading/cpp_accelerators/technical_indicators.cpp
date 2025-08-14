#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>

extern "C" {

double calculate_rsi(const double* prices, int length, int period) {
    if (length < period + 1) return 50.0;
    
    double gain_sum = 0.0, loss_sum = 0.0;
    
    for (int i = 1; i <= period; ++i) {
        double change = prices[length - i] - prices[length - i - 1];
        if (change > 0) gain_sum += change;
        else loss_sum += -change;
    }
    
    double avg_gain = gain_sum / period;
    double avg_loss = loss_sum / period;
    
    if (avg_loss == 0) return 100.0;
    
    double rs = avg_gain / avg_loss;
    return 100.0 - (100.0 / (1.0 + rs));
}

double calculate_macd(const double* prices, int length, int fast, int slow) {
    if (length < slow) return 0.0;
    
    double fast_multiplier = 2.0 / (fast + 1);
    double slow_multiplier = 2.0 / (slow + 1);
    
    double fast_ema = prices[length - slow];
    double slow_ema = prices[length - slow];
    
    for (int i = length - slow + 1; i < length; ++i) {
        fast_ema = (prices[i] - fast_ema) * fast_multiplier + fast_ema;
        slow_ema = (prices[i] - slow_ema) * slow_multiplier + slow_ema;
    }
    
    return fast_ema - slow_ema;
}

double calculate_bollinger_std(const double* prices, int length, int period) {
    if (length < period) return 0.0;
    
    double sum = 0.0;
    for (int i = length - period; i < length; ++i) {
        sum += prices[i];
    }
    double mean = sum / period;
    
    double variance = 0.0;
    for (int i = length - period; i < length; ++i) {
        variance += (prices[i] - mean) * (prices[i] - mean);
    }
    
    return std::sqrt(variance / period);
}

double calculate_atr(const double* highs, const double* lows, const double* closes, int length, int period) {
    if (length < period + 1) return 0.0;
    
    double atr_sum = 0.0;
    
    for (int i = length - period; i < length; ++i) {
        double tr1 = highs[i] - lows[i];
        double tr2 = std::abs(highs[i] - closes[i - 1]);
        double tr3 = std::abs(lows[i] - closes[i - 1]);
        double true_range = std::max({tr1, tr2, tr3});
        atr_sum += true_range;
    }
    
    return atr_sum / period;
}

double calculate_stochastic_k(const double* highs, const double* lows, const double* closes, int length, int period) {
    if (length < period) return 50.0;
    
    double highest = *std::max_element(highs + length - period, highs + length);
    double lowest = *std::min_element(lows + length - period, lows + length);
    
    if (highest == lowest) return 50.0;
    
    return ((closes[length - 1] - lowest) / (highest - lowest)) * 100.0;
}

}
