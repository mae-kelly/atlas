#include <vector>
#include <algorithm>
#include <cmath>

extern "C" {

double detect_liquidity_crisis(const double* bid_ask_spreads, const double* volumes, int length) {
    if (length < 10) return 0.0;
    
    double spread_mean = 0.0, volume_mean = 0.0;
    for (int i = 0; i < length; ++i) {
        spread_mean += bid_ask_spreads[i];
        volume_mean += volumes[i];
    }
    spread_mean /= length;
    volume_mean /= length;
    
    double spread_std = 0.0, volume_std = 0.0;
    for (int i = 0; i < length; ++i) {
        spread_std += (bid_ask_spreads[i] - spread_mean) * (bid_ask_spreads[i] - spread_mean);
        volume_std += (volumes[i] - volume_mean) * (volumes[i] - volume_mean);
    }
    spread_std = std::sqrt(spread_std / (length - 1));
    volume_std = std::sqrt(volume_std / (length - 1));
    
    double current_spread = bid_ask_spreads[length - 1];
    double current_volume = volumes[length - 1];
    
    double spread_z = (current_spread - spread_mean) / spread_std;
    double volume_z = (volume_mean - current_volume) / volume_std;
    
    return (spread_z + volume_z) / 2.0;
}

double calculate_market_impact(double order_size, const double* volumes, int length) {
    if (length < 1) return 0.0;
    
    double avg_volume = 0.0;
    for (int i = 0; i < length; ++i) {
        avg_volume += volumes[i];
    }
    avg_volume /= length;
    
    if (avg_volume == 0.0) return 1.0;
    
    double participation_rate = order_size / avg_volume;
    return std::pow(participation_rate, 0.5) * 0.1;
}

double estimate_slippage(double order_size, const double* bids, const double* asks, const double* sizes, int depth) {
    if (depth < 1 || order_size <= 0.0) return 0.0;
    
    double remaining_size = order_size;
    double weighted_price = 0.0;
    double total_filled = 0.0;
    
    for (int i = 0; i < depth && remaining_size > 0.0; ++i) {
        double fill_size = std::min(remaining_size, sizes[i]);
        weighted_price += asks[i] * fill_size;
        total_filled += fill_size;
        remaining_size -= fill_size;
    }
    
    if (total_filled == 0.0) return 0.0;
    
    double avg_fill_price = weighted_price / total_filled;
    double mid_price = (bids[0] + asks[0]) / 2.0;
    
    return (avg_fill_price - mid_price) / mid_price;
}

}
