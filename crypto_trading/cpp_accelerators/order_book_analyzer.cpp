#include <vector>
#include <algorithm>
#include <cmath>

extern "C" {

double calculate_bid_ask_spread_impact(const double* bids, const double* asks, const double* bid_sizes, const double* ask_sizes, int depth) {
    if (depth < 1) return 0.0;
    
    double weighted_bid = 0.0, weighted_ask = 0.0;
    double total_bid_size = 0.0, total_ask_size = 0.0;
    
    for (int i = 0; i < depth; ++i) {
        weighted_bid += bids[i] * bid_sizes[i];
        weighted_ask += asks[i] * ask_sizes[i];
        total_bid_size += bid_sizes[i];
        total_ask_size += ask_sizes[i];
    }
    
    if (total_bid_size == 0.0 || total_ask_size == 0.0) return 0.0;
    
    double avg_bid = weighted_bid / total_bid_size;
    double avg_ask = weighted_ask / total_ask_size;
    
    return (avg_ask - avg_bid) / ((avg_ask + avg_bid) / 2.0);
}

double calculate_order_flow_imbalance(const double* buy_volume, const double* sell_volume, int length) {
    if (length < 1) return 0.0;
    
    double total_buy = 0.0, total_sell = 0.0;
    
    for (int i = 0; i < length; ++i) {
        total_buy += buy_volume[i];
        total_sell += sell_volume[i];
    }
    
    double total_volume = total_buy + total_sell;
    if (total_volume == 0.0) return 0.0;
    
    return (total_buy - total_sell) / total_volume;
}

double calculate_vwap(const double* prices, const double* volumes, int length) {
    if (length < 1) return 0.0;
    
    double price_volume_sum = 0.0;
    double volume_sum = 0.0;
    
    for (int i = 0; i < length; ++i) {
        price_volume_sum += prices[i] * volumes[i];
        volume_sum += volumes[i];
    }
    
    if (volume_sum == 0.0) return 0.0;
    
    return price_volume_sum / volume_sum;
}

double calculate_twap(const double* prices, int length) {
    if (length < 1) return 0.0;
    
    double sum = 0.0;
    for (int i = 0; i < length; ++i) {
        sum += prices[i];
    }
    
    return sum / length;
}

}
