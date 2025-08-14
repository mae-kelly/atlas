use std::collections::VecDeque;
use anyhow::Result;

use crate::core::types::*;

pub struct SignalGenerator {
    price_history: VecDeque<f64>,
    volume_history: VecDeque<f64>,
    signal_history: VecDeque<f64>,
}

impl SignalGenerator {
    pub fn new() -> Self {
        Self {
            price_history: VecDeque::with_capacity(200),
            volume_history: VecDeque::with_capacity(200),
            signal_history: VecDeque::with_capacity(50),
        }
    }
    
    pub fn update(&mut self, price: f64, volume: f64) {
        self.price_history.push_back(price);
        self.volume_history.push_back(volume);
        
        if self.price_history.len() > 200 {
            self.price_history.pop_front();
        }
        if self.volume_history.len() > 200 {
            self.volume_history.pop_front();
        }
    }
    
    pub fn calculate_momentum_signal(&self) -> f64 {
        if self.price_history.len() < 20 {
            return 0.0;
        }
        
        let recent_prices: Vec<f64> = self.price_history.iter().rev().take(20).cloned().collect();
        let short_ma = recent_prices.iter().take(5).sum::<f64>() / 5.0;
        let long_ma = recent_prices.iter().sum::<f64>() / 20.0;
        
        (short_ma - long_ma) / long_ma
    }
    
    pub fn calculate_volume_signal(&self) -> f64 {
        if self.volume_history.len() < 20 {
            return 0.0;
        }
        
        let recent_volumes: Vec<f64> = self.volume_history.iter().rev().take(20).cloned().collect();
        let current_volume = recent_volumes[0];
        let avg_volume = recent_volumes.iter().sum::<f64>() / 20.0;
        
        if avg_volume > 0.0 {
            (current_volume - avg_volume) / avg_volume
        } else {
            0.0
        }
    }
    
    pub fn calculate_rsi(&self, period: usize) -> f64 {
        if self.price_history.len() < period + 1 {
            return 50.0;
        }
        
        let prices: Vec<f64> = self.price_history.iter().rev().take(period + 1).cloned().collect();
        let mut gains = 0.0;
        let mut losses = 0.0;
        
        for i in 1..prices.len() {
            let change = prices[i-1] - prices[i];
            if change > 0.0 {
                gains += change;
            } else {
                losses += -change;
            }
        }
        
        let avg_gain = gains / period as f64;
        let avg_loss = losses / period as f64;
        
        if avg_loss == 0.0 {
            return 100.0;
        }
        
        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }
    
    pub fn calculate_bollinger_bands(&self, period: usize, std_dev: f64) -> (f64, f64, f64) {
        if self.price_history.len() < period {
            let current_price = self.price_history.back().unwrap_or(&0.0);
            return (*current_price, *current_price, *current_price);
        }
        
        let prices: Vec<f64> = self.price_history.iter().rev().take(period).cloned().collect();
        let sma = prices.iter().sum::<f64>() / period as f64;
        
        let variance = prices.iter()
            .map(|price| (price - sma).powi(2))
            .sum::<f64>() / period as f64;
        
        let std = variance.sqrt();
        
        let upper = sma + (std * std_dev);
        let lower = sma - (std * std_dev);
        
        (upper, sma, lower)
    }
    
    pub fn calculate_macd(&self, fast: usize, slow: usize, signal: usize) -> (f64, f64, f64) {
        if self.price_history.len() < slow {
            return (0.0, 0.0, 0.0);
        }
        
        let prices: Vec<f64> = self.price_history.iter().cloned().collect();
        
        let fast_ema = self.calculate_ema(&prices, fast);
        let slow_ema = self.calculate_ema(&prices, slow);
        let macd_line = fast_ema - slow_ema;
        
        let signal_line = macd_line;
        let histogram = macd_line - signal_line;
        
        (macd_line, signal_line, histogram)
    }
    
    fn calculate_ema(&self, prices: &[f64], period: usize) -> f64 {
        if prices.is_empty() {
            return 0.0;
        }
        
        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut ema = prices[0];
        
        for &price in prices.iter().skip(1) {
            ema = (price - ema) * multiplier + ema;
        }
        
        ema
    }
}
