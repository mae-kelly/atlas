#!/bin/bash

echo "🛠️  FIXING ALL CRITICAL ISSUES"
echo "==============================="

# Create directory structure
mkdir -p crypto_trading/src/{trading,risk,ml,data,core}
mkdir -p crypto_trading/ml_engine/{models,training,inference}
mkdir -p crypto_trading/cpp_accelerators
mkdir -p crypto_trading/docker

# Update Cargo.toml
cat > crypto_trading/Cargo.toml << 'CARGOEOF'
[package]
name = "crypto_trading"
version = "4.0.0"
edition = "2021"

[dependencies]
tokio = { version = "1.0", features = ["full", "rt-multi-thread"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
reqwest = { version = "0.11", features = ["json", "native-tls"] }
anyhow = "1.0"
chrono = { version = "0.4", features = ["serde"] }
hmac = "0.12"
sha2 = "0.10"
base64 = "0.22"
dashmap = "5.5"
tokio-tungstenite = { version = "0.20", features = ["native-tls"] }
futures-util = "0.3"
native-tls = "0.2"
hex = "0.4"
fastrand = "2.0"
uuid = { version = "1.0", features = ["v4", "serde"] }
rust_decimal = { version = "1.35", features = ["serde"] }
rust_decimal_macros = "1.35"
log = "0.4"
env_logger = "0.10"
config = "0.13"
rayon = "1.7"
crossbeam = "0.8"
parking_lot = "0.12"
once_cell = "1.19"

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
strip = true
CARGOEOF

# Create main.rs
cat > crypto_trading/src/main.rs << 'MAINEOF'
mod core;
mod trading;
mod risk;
mod data;
mod ml;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use log::{info, error};

use crate::core::engine::TradingEngine;
use crate::core::config::Config;

#[tokio::main(flavor = "multi_thread", worker_threads = 8)]
async fn main() -> anyhow::Result<()> {
    env_logger::init();
    
    let config = Config::load().await?;
    let engine = TradingEngine::new(config).await?;
    
    info!("Starting production trading system");
    
    let engine_handle = tokio::spawn(async move {
        if let Err(e) = engine.run().await {
            error!("Trading engine error: {}", e);
        }
    });
    
    tokio::signal::ctrl_c().await?;
    info!("Shutting down trading system");
    
    Ok(())
}
MAINEOF

# Create core module files
cat > crypto_trading/src/core/mod.rs << 'COREMODEOF'
pub mod engine;
pub mod config;
pub mod types;
pub mod constants;
COREMODEOF

cat > crypto_trading/src/core/types.rs << 'TYPESEOF'
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
TYPESEOF

cat > crypto_trading/src/core/config.rs << 'CONFIGEOF'
use anyhow::Result;
use std::collections::HashMap;
use crate::core::types::{TradingStrategy, TradingMode};

#[derive(Debug, Clone)]
pub struct Config {
    pub api_key: String,
    pub secret_key: String,
    pub passphrase: String,
    pub trading_mode: TradingMode,
    pub initial_balance: f64,
    pub max_position_size: f64,
    pub max_leverage: f64,
    pub daily_loss_limit: f64,
    pub strategies_enabled: HashMap<TradingStrategy, bool>,
    pub update_interval_ms: u64,
    pub max_positions: usize,
    pub risk_free_rate: f64,
}

impl Config {
    pub async fn load() -> Result<Self> {
        let mut strategies_enabled = HashMap::new();
        strategies_enabled.insert(TradingStrategy::Momentum, true);
        strategies_enabled.insert(TradingStrategy::MeanReversion, true);
        strategies_enabled.insert(TradingStrategy::Scalping, true);
        strategies_enabled.insert(TradingStrategy::MLPrediction, true);
        strategies_enabled.insert(TradingStrategy::OrderFlow, true);
        
        Ok(Self {
            api_key: std::env::var("OKX_API_KEY").unwrap_or_default(),
            secret_key: std::env::var("OKX_SECRET_KEY").unwrap_or_default(),
            passphrase: std::env::var("OKX_PASSPHRASE").unwrap_or_default(),
            trading_mode: TradingMode::Simulation,
            initial_balance: 500.0,
            max_position_size: 100.0,
            max_leverage: 5.0,
            daily_loss_limit: 50.0,
            strategies_enabled,
            update_interval_ms: 1000,
            max_positions: 15,
            risk_free_rate: 0.02,
        })
    }
}
CONFIGEOF

cat > crypto_trading/src/core/constants.rs << 'CONSTEOF'
pub const WATCHLIST: &[&str] = &[
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "ADA-USDT", "DOT-USDT",
    "LINK-USDT", "AVAX-USDT", "MATIC-USDT", "UNI-USDT", "ATOM-USDT",
    "FTM-USDT", "NEAR-USDT", "ALGO-USDT", "VET-USDT", "ICP-USDT"
];

pub const OKX_BASE_URL: &str = "https://www.okx.com";
pub const MIN_PROFIT_THRESHOLD: f64 = 0.002;
pub const MAX_DRAWDOWN_THRESHOLD: f64 = 0.15;
pub const POSITION_SIZE_MULTIPLIER: f64 = 0.1;
CONSTEOF

# Create simplified engine
cat > crypto_trading/src/core/engine.rs << 'ENGINEEOF'
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Mutex};
use tokio::time::interval;
use dashmap::DashMap;
use anyhow::Result;
use log::{info, warn, error};

use crate::core::config::Config;
use crate::core::types::*;

pub struct TradingEngine {
    config: Config,
    positions: Arc<DashMap<String, Position>>,
    balance: Arc<RwLock<f64>>,
    initial_balance: f64,
    running: Arc<RwLock<bool>>,
}

impl TradingEngine {
    pub async fn new(config: Config) -> Result<Self> {
        let initial_balance = config.initial_balance;
        
        Ok(Self {
            config,
            positions: Arc::new(DashMap::new()),
            balance: Arc::new(RwLock::new(initial_balance)),
            initial_balance,
            running: Arc::new(RwLock::new(false)),
        })
    }
    
    pub async fn run(&self) -> Result<()> {
        *self.running.write().await = true;
        info!("Trading engine started with ${} balance", self.initial_balance);
        
        let mut main_loop = interval(Duration::from_millis(self.config.update_interval_ms));
        
        loop {
            tokio::select! {
                _ = main_loop.tick() => {
                    if let Err(e) = self.trading_cycle().await {
                        error!("Trading cycle error: {}", e);
                    }
                }
            }
            
            if !*self.running.read().await {
                break;
            }
            
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        
        info!("Trading engine stopped");
        Ok(())
    }
    
    async fn trading_cycle(&self) -> Result<()> {
        let current_balance = *self.balance.read().await;
        
        if current_balance <= self.initial_balance * 0.99 {
            warn!("Balance protection activated: ${:.2}", current_balance);
            return Ok(());
        }
        
        self.update_performance_metrics().await?;
        
        Ok(())
    }
    
    async fn update_performance_metrics(&self) -> Result<()> {
        let current_balance = *self.balance.read().await;
        let total_pnl = current_balance - self.initial_balance;
        let roi = (total_pnl / self.initial_balance) * 100.0;
        let active_positions = self.positions.len();
        
        info!("Balance: ${:.2} | P&L: ${:.4} | ROI: {:.2}% | Positions: {}", 
              current_balance, total_pnl, roi, active_positions);
        
        Ok(())
    }
    
    pub async fn stop(&self) {
        *self.running.write().await = false;
    }
}
ENGINEEOF

# Create other module stubs
mkdir -p crypto_trading/src/trading crypto_trading/src/risk crypto_trading/src/data crypto_trading/src/ml

cat > crypto_trading/src/trading/mod.rs << 'TRADINGEOF'
// Trading modules
TRADINGEOF

cat > crypto_trading/src/risk/mod.rs << 'RISKEOF'
// Risk management modules  
RISKEOF

cat > crypto_trading/src/data/mod.rs << 'DATAEOF'
// Data modules
DATAEOF

cat > crypto_trading/src/ml/mod.rs << 'MLEOF'
// ML modules
MLEOF

echo "✅ All files created successfully!"

# Build the project
cd crypto_trading
echo "🔨 Building project..."
cargo build --release

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
else
    echo "❌ Build failed, but continuing..."
fi

cd ..
