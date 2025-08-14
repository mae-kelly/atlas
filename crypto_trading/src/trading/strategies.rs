use std::collections::HashMap;
use anyhow::Result;
use log::{info, debug};

use crate::core::config::Config;
use crate::core::types::*;
use crate::core::engine::TradingSignal;

pub struct StrategyManager {
    config: Config,
    momentum_strategy: MomentumStrategy,
    mean_reversion_strategy: MeanReversionStrategy,
    scalping_strategy: ScalpingStrategy,
    ml_strategy: MLStrategy,
}

impl StrategyManager {
    pub async fn new(config: &Config) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            momentum_strategy: MomentumStrategy::new(),
            mean_reversion_strategy: MeanReversionStrategy::new(),
            scalping_strategy: ScalpingStrategy::new(),
            ml_strategy: MLStrategy::new(),
        })
    }
    
    pub async fn generate_signals(
        &self,
        market_data: &HashMap<String, MarketData>,
        ml_predictions: &HashMap<String, f64>,
    ) -> Result<Vec<TradingSignal>> {
        let mut signals = Vec::new();
        
        for (symbol, data) in market_data {
            if let Some(indicators) = self.calculate_indicators(data) {
                if self.config.strategies_enabled.get(&TradingStrategy::Momentum).unwrap_or(&false) {
                    if let Some(signal) = self.momentum_strategy.generate_signal(symbol, data, &indicators).await {
                        signals.push(signal);
                    }
                }
                
                if self.config.strategies_enabled.get(&TradingStrategy::MeanReversion).unwrap_or(&false) {
                    if let Some(signal) = self.mean_reversion_strategy.generate_signal(symbol, data, &indicators).await {
                        signals.push(signal);
                    }
                }
                
                if self.config.strategies_enabled.get(&TradingStrategy::Scalping).unwrap_or(&false) {
                    if let Some(signal) = self.scalping_strategy.generate_signal(symbol, data, &indicators).await {
                        signals.push(signal);
                    }
                }
                
                if self.config.strategies_enabled.get(&TradingStrategy::MLPrediction).unwrap_or(&false) {
                    if let Some(prediction) = ml_predictions.get(symbol) {
                        if let Some(signal) = self.ml_strategy.generate_signal(symbol, data, &indicators, *prediction).await {
                            signals.push(signal);
                        }
                    }
                }
            }
        }
        
        Ok(signals)
    }
    
    fn calculate_indicators(&self, data: &MarketData) -> Option<TechnicalIndicators> {
        Some(TechnicalIndicators {
            rsi: 50.0,
            macd: 0.0,
            bb_upper: data.price * 1.02,
            bb_lower: data.price * 0.98,
            atr: data.price * 0.01,
            volume_sma: data.volume,
            price_momentum: 0.0,
            liquidity_score: 0.8,
        })
    }
}

struct MomentumStrategy;

impl MomentumStrategy {
    fn new() -> Self {
        Self
    }
    
    async fn generate_signal(
        &self,
        symbol: &str,
        data: &MarketData,
        indicators: &TechnicalIndicators,
    ) -> Option<TradingSignal> {
        if indicators.rsi > 70.0 && indicators.macd > 0.0 {
            return Some(TradingSignal {
                symbol: symbol.to_string(),
                side: "sell".to_string(),
                price: data.price,
                confidence: 0.7,
                strategy: TradingStrategy::Momentum,
                stop_loss_pct: 0.02,
                take_profit_pct: 0.04,
                timestamp: data.timestamp,
            });
        }
        
        if indicators.rsi < 30.0 && indicators.macd < 0.
