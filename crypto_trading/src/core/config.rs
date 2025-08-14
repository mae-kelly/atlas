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
