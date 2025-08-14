#pragma once
#include <vector>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace HFT {

struct TickData {
    std::string symbol;
    double price;
    double bid;
    double ask;
    double volume;
    uint64_t timestamp_ns;
    double spread_bps;
    double momentum_1s;
    double momentum_5s;
    double velocity;
    uint32_t tick_count;
};

struct Signal {
    std::string symbol;
    std::string strategy;
    std::string side;
    double confidence;
    double entry_price;
    double target_price;
    double stop_price;
    uint64_t timestamp_ns;
    double expected_pnl;
};

struct Position {
    std::string id;
    std::string symbol;
    std::string side;
    double entry_price;
    double current_price;
    double quantity;
    uint64_t entry_time_ns;
    double unrealized_pnl;
    bool is_open;
};

class UltraFastProcessor {
private:
    std::unordered_map<std::string, TickData> latest_ticks_;
    std::unordered_map<std::string, std::vector<double>> price_buffers_;
    std::vector<Signal> pending_signals_;
    std::vector<Position> active_positions_;
    
    std::atomic<uint64_t> total_ticks_{0};
    std::atomic<uint64_t> signals_generated_{0};
    std::atomic<bool> running_{false};
    
    mutable std::mutex data_mutex_;
    mutable std::mutex signal_mutex_;
    
public:
    void ProcessTick(const TickData& tick);
    std::vector<Signal> GenerateSignals();
    void UpdatePositions();
    
    // Strategy functions
    Signal MomentumStrategy(const TickData& tick);
    Signal ScalpingStrategy(const TickData& tick);
    Signal MeanReversionStrategy(const TickData& tick);
    Signal VolumeStrategy(const TickData& tick);
    Signal ArbitrageStrategy(const TickData& tick);
    Signal MLPredictionStrategy(const TickData& tick, double ml_score);
    
    // Performance functions
    double CalculateVelocity(const std::string& symbol);
    double CalculateMomentum(const std::string& symbol, int periods);
    double CalculateVolatility(const std::string& symbol);
    
    uint64_t GetTickCount() const { return total_ticks_.load(); }
    uint64_t GetSignalCount() const { return signals_generated_.load(); }
    std::vector<TickData> GetTopMovers() const;
};

}
