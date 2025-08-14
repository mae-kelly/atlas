use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TradingStrategy {
    Momentum,
    MeanReversion,
    Scalping,
    MarketMaking,
    Arbitrage,
    MLPrediction,
    OrderFlow,
    VolumeProfile,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradingMode {
    Simulation,
    PaperTrading,
    LiveTrading,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarketCondition {
    Trending,
    Sideways,
    Volatile,
    LowVolume,
    Crisis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub id: String,
    pub symbol: String,
    pub strategy: TradingStrategy,
    pub entry_price: f64,
    pub current_price: f64,
    pub quantity: f64,
    pub leverage: f64,
    pub side: String,
    pub entry_time: Instant,
    pub stop_loss: f64,
    pub take_profit: f64,
    pub unrealized_pnl: f64,
    pub fees_paid: f64,
    pub order_id: Option<String>,
    pub is_simulated: bool,
}

#[derive(Debug, Clone)]
pub struct MarketData {
    pub symbol: String,
    pub price: f64,
    pub volume: f64,
    pub bid: f64,
    pub ask: f64,
    pub timestamp: u64,
    pub order_book: Option<OrderBook>,
}

#[derive(Debug, Clone)]
pub struct OrderBook {
    pub bids: Vec<(f64, f64)>,
    pub asks: Vec<(f64, f64)>,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub struct TechnicalIndicators {
    pub rsi: f64,
    pub macd: f64,
    pub bb_upper: f64,
    pub bb_lower: f64,
    pub atr: f64,
    pub volume_sma: f64,
    pub price_momentum: f64,
    pub liquidity_score: f64,
}

#[derive(Debug, Clone)]
pub struct RiskMetrics {
    pub var_95: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub portfolio_beta: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
}
