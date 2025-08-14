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
