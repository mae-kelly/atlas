#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>

extern "C" {

double calculate_portfolio_var(const double* returns, int length, double confidence) {
    if (length < 2) return 0.0;
    
    std::vector<double> sorted_returns(returns, returns + length);
    std::sort(sorted_returns.begin(), sorted_returns.end());
    
    int index = static_cast<int>((1.0 - confidence) * length);
    return -sorted_returns[index];
}

double calculate_sharpe_ratio(const double* returns, int length, double risk_free_rate) {
    if (length < 2) return 0.0;
    
    double mean_return = std::accumulate(returns, returns + length, 0.0) / length;
    
    double variance = 0.0;
    for (int i = 0; i < length; ++i) {
        variance += (returns[i] - mean_return) * (returns[i] - mean_return);
    }
    double std_dev = std::sqrt(variance / (length - 1));
    
    if (std_dev == 0.0) return 0.0;
    
    return (mean_return - risk_free_rate) / std_dev;
}

double calculate_max_drawdown(const double* equity_curve, int length) {
    if (length < 2) return 0.0;
    
    double max_dd = 0.0;
    double peak = equity_curve[0];
    
    for (int i = 1; i < length; ++i) {
        if (equity_curve[i] > peak) {
            peak = equity_curve[i];
        } else {
            double drawdown = (peak - equity_curve[i]) / peak;
            max_dd = std::max(max_dd, drawdown);
        }
    }
    
    return max_dd;
}

double calculate_beta(const double* asset_returns, const double* market_returns, int length) {
    if (length < 2) return 1.0;
    
    double asset_mean = std::accumulate(asset_returns, asset_returns + length, 0.0) / length;
    double market_mean = std::accumulate(market_returns, market_returns + length, 0.0) / length;
    
    double covariance = 0.0;
    double market_variance = 0.0;
    
    for (int i = 0; i < length; ++i) {
        covariance += (asset_returns[i] - asset_mean) * (market_returns[i] - market_mean);
        market_variance += (market_returns[i] - market_mean) * (market_returns[i] - market_mean);
    }
    
    if (market_variance == 0.0) return 1.0;
    
    return covariance / market_variance;
}

}
