#include "hft_core.hpp"
#include <algorithm>
#include <cmath>
#include <random>

namespace HFT {

void UltraFastProcessor::ProcessTick(const TickData& tick) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    // Update price buffer
    auto& buffer = price_buffers_[tick.symbol];
    buffer.push_back(tick.price);
    if (buffer.size() > 1000) {
        buffer.erase(buffer.begin());
    }
    
    // Calculate derived metrics
    TickData enhanced_tick = tick;
    enhanced_tick.velocity = CalculateVelocity(tick.symbol);
    enhanced_tick.momentum_1s = CalculateMomentum(tick.symbol, 20);
    enhanced_tick.momentum_5s = CalculateMomentum(tick.symbol, 100);
    enhanced_tick.spread_bps = ((tick.ask - tick.bid) / tick.bid) * 10000.0;
    
    latest_ticks_[tick.symbol] = enhanced_tick;
    total_ticks_.fetch_add(1);
}

std::vector<Signal> UltraFastProcessor::GenerateSignals() {
    std::vector<Signal> signals;
    std::lock_guard<std::mutex> lock(signal_mutex_);
    
    for (const auto& [symbol, tick] : latest_ticks_) {
        // Try each strategy
        auto momentum_signal = MomentumStrategy(tick);
        if (momentum_signal.confidence > 0.7) {
            signals.push_back(momentum_signal);
        }
        
        auto scalping_signal = ScalpingStrategy(tick);
        if (scalping_signal.confidence > 0.6) {
            signals.push_back(scalping_signal);
        }
        
        auto reversion_signal = MeanReversionStrategy(tick);
        if (reversion_signal.confidence > 0.65) {
            signals.push_back(reversion_signal);
        }
        
        auto volume_signal = VolumeStrategy(tick);
        if (volume_signal.confidence > 0.75) {
            signals.push_back(volume_signal);
        }
    }
    
    signals_generated_.fetch_add(signals.size());
    return signals;
}

Signal UltraFastProcessor::MomentumStrategy(const TickData& tick) {
    Signal signal{};
    signal.symbol = tick.symbol;
    signal.strategy = "MOMENTUM";
    
    if (std::abs(tick.momentum_1s) > 5.0 && std::abs(tick.velocity) > 0.1) {
        signal.confidence = std::min(1.0, std::abs(tick.momentum_1s) / 20.0);
        signal.side = tick.momentum_1s > 0 ? "BUY" : "SELL";
        signal.entry_price = tick.side == "BUY" ? tick.ask : tick.bid;
        signal.target_price = signal.side == "BUY" ? 
            signal.entry_price * 1.003 : signal.entry_price * 0.997;
        signal.stop_price = signal.side == "BUY" ? 
            signal.entry_price * 0.999 : signal.entry_price * 1.001;
        signal.timestamp_ns = tick.timestamp_ns;
    }
    
    return signal;
}

Signal UltraFastProcessor::ScalpingStrategy(const TickData& tick) {
    Signal signal{};
    signal.symbol = tick.symbol;
    signal.strategy = "SCALPING";
    
    if (tick.spread_bps > 2.0 && tick.spread_bps < 15.0 && tick.volume > 100000.0) {
        signal.confidence = std::min(1.0, tick.spread_bps / 10.0);
        signal.side = "BUY";
        signal.entry_price = (tick.bid + tick.ask) / 2.0;
        signal.target_price = signal.entry_price * 1.0015;
        signal.stop_price = signal.entry_price * 0.9995;
        signal.timestamp_ns = tick.timestamp_ns;
    }
    
    return signal;
}

Signal UltraFastProcessor::MeanReversionStrategy(const TickData& tick) {
    Signal signal{};
    signal.symbol = tick.symbol;
    signal.strategy = "REVERSION";
    
    if (std::abs(tick.momentum_5s) > 15.0 && std::abs(tick.momentum_1s) < 3.0) {
        signal.confidence = std::min(1.0, std::abs(tick.momentum_5s) / 30.0);
        signal.side = tick.momentum_5s > 0 ? "SELL" : "BUY";
        signal.entry_price = signal.side == "BUY" ? tick.ask : tick.bid;
        signal.target_price = signal.side == "BUY" ? 
            signal.entry_price * 1.005 : signal.entry_price * 0.995;
        signal.stop_price = signal.side == "BUY" ? 
            signal.entry_price * 0.997 : signal.entry_price * 1.003;
        signal.timestamp_ns = tick.timestamp_ns;
    }
    
    return signal;
}

Signal UltraFastProcessor::VolumeStrategy(const TickData& tick) {
    Signal signal{};
    signal.symbol = tick.symbol;
    signal.strategy = "VOLUME";
    
    // Volume spike detection would need historical volume data
    if (tick.volume > 1000000.0 && std::abs(tick.momentum_1s) > 3.0) {
        signal.confidence = 0.8;
        signal.side = tick.momentum_1s > 0 ? "BUY" : "SELL";
        signal.entry_price = signal.side == "BUY" ? tick.ask : tick.bid;
        signal.target_price = signal.side == "BUY" ? 
            signal.entry_price * 1.004 : signal.entry_price * 0.996;
        signal.stop_price = signal.side == "BUY" ? 
            signal.entry_price * 0.998 : signal.entry_price * 1.002;
        signal.timestamp_ns = tick.timestamp_ns;
    }
    
    return signal;
}

double UltraFastProcessor::CalculateVelocity(const std::string& symbol) {
    auto it = price_buffers_.find(symbol);
    if (it == price_buffers_.end() || it->second.size() < 2) {
        return 0.0;
    }
    
    const auto& buffer = it->second;
    double current = buffer.back();
    double previous = buffer[buffer.size() - 2];
    
    return (current - previous) / previous * 10000.0; // basis points per tick
}

double UltraFastProcessor::CalculateMomentum(const std::string& symbol, int periods) {
    auto it = price_buffers_.find(symbol);
    if (it == price_buffers_.end() || it->second.size() < periods) {
        return 0.0;
    }
    
    const auto& buffer = it->second;
    double current = buffer.back();
    double past = buffer[buffer.size() - periods];
    
    return (current - past) / past * 10000.0; // basis points
}

std::vector<TickData> UltraFastProcessor::GetTopMovers() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    std::vector<TickData> movers;
    for (const auto& [symbol, tick] : latest_ticks_) {
        if (std::abs(tick.momentum_1s) > 2.0) {
            movers.push_back(tick);
        }
    }
    
    std::sort(movers.begin(), movers.end(), 
        [](const TickData& a, const TickData& b) {
            return std::abs(a.momentum_1s) > std::abs(b.momentum_1s);
        });
    
    return movers;
}

}
