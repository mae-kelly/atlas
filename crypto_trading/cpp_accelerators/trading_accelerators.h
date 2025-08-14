#ifndef TRADING_ACCELERATORS_H
#define TRADING_ACCELERATORS_H

extern "C" {
    double calculate_rsi(const double* prices, int length, int period);
    double calculate_macd(const double* prices, int length, int fast, int slow);
    double calculate_bollinger_std(const double* prices, int length, int period);
    double calculate_atr(const double* highs, const double* lows, const double* closes, int length, int period);
    double calculate_stochastic_k(const double* highs, const double* lows, const double* closes, int length, int period);
    
    double calculate_portfolio_var(const double* returns, int length, double confidence);
    double calculate_sharpe_ratio(const double* returns, int length, double risk_free_rate);
    double calculate_max_drawdown(const double* equity_curve, int length);
    double calculate_beta(const double* asset_returns, const double* market_returns, int length);
    
    double calculate_bid_ask_spread_impact(const double* bids, const double* asks, const double* bid_sizes, const double* ask_sizes, int depth);
    double calculate_order_flow_imbalance(const double* buy_volume, const double* sell_volume, int length);
    double calculate_vwap(const double* prices, const double* volumes, int length);
    double calculate_twap(const double* prices, int length);
    
    double detect_liquidity_crisis(const double* bid_ask_spreads, const double* volumes, int length);
    double calculate_market_impact(double order_size, const double* volumes, int length);
    double estimate_slippage(double order_size, const double* bids, const double* asks, const double* sizes, int depth);
}

#endif
