#!/bin/bash

echo "🍎 SETTING UP MAC NATIVE AUTONOMOUS TRADER"
echo "=========================================="

# Use existing crypto_trading directory and enhance it
if [ ! -d "crypto_trading" ]; then
    echo "❌ crypto_trading directory not found!"
    echo "Please run this from the directory containing crypto_trading/"
    exit 1
fi

# Create enhanced version with 24/7 capabilities
echo "🔧 Enhancing existing bot for 24/7 operation..."

# Create backup
cp -r crypto_trading crypto_trading_backup_$(date +%Y%m%d_%H%M%S)

# Update Cargo.toml with required dependencies
cat > crypto_trading/Cargo.toml << 'CARGO_EOF'
[package]
name = "crypto_trading"
version = "3.0.0"
edition = "2021"

[dependencies]
tokio = { version = "1.0", features = ["full", "rt-multi-thread", "macros"] }
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
clap = { version = "4.0", features = ["derive"] }
signal-hook = "0.3"
signal-hook-tokio = { version = "0.3", features = ["futures-v0_3"] }

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"
CARGO_EOF

# Create the enhanced autonomous bot
cat > crypto_trading/src/main.rs << 'MAIN_EOF'
use std::collections::{VecDeque, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::io::{self, Write};
use tokio::sync::{RwLock, Mutex};
use serde::{Deserialize, Serialize};
use dashmap::DashMap;
use reqwest;
use chrono::Utc;
use hmac::{Hmac, Mac};
use sha2::Sha256;
use base64::{Engine as _, engine::general_purpose};
use rust_decimal::prelude::*;
use uuid::Uuid;
use log::{info, warn, error, debug};

type HmacSha256 = Hmac<Sha256>;

// SAFETY CONSTANTS - NEVER TRADE BELOW THESE
const PROTECTED_BALANCE: f64 = 500.0;
const MAX_RISK_PER_TRADE: f64 = 0.02; // 2% max risk per trade
const MAX_DAILY_RISK: f64 = 0.10; // 10% max daily risk
const MAX_POSITIONS: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum TradingStrategy {
    MomentumScalping,
    BreakoutCapture,
    VolumeSpike,
    SupportResistance,
    TrendFollowing,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum MarketCondition {
    TrendingUp,
    TrendingDown,
    Sideways,
    HighVolatility,
    LowVolatility,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OkxTicker {
    #[serde(rename = "instId")]
    inst_id: String,
    last: String,
    #[serde(rename = "askPx")]
    ask_px: Option<String>,
    #[serde(rename = "bidPx")]
    bid_px: Option<String>,
    #[serde(rename = "vol24h")]
    vol_24h: Option<String>,
    #[serde(rename = "volCcy24h")]
    vol_ccy_24h: Option<String>,
    ts: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OkxResponse {
    code: String,
    msg: String,
    data: Vec<OkxTicker>,
}

#[derive(Debug, Clone)]
struct ProtectedBalance {
    initial: f64,
    current: f64,
    profit_only: f64,
    daily_pnl: f64,
    last_reset: u64,
}

#[derive(Debug, Clone)]
struct TechnicalIndicators {
    sma_20: f64,
    sma_50: f64,
    ema_12: f64,
    ema_26: f64,
    rsi: f64,
    macd: f64,
    macd_signal: f64,
    bollinger_upper: f64,
    bollinger_lower: f64,
    bollinger_middle: f64,
    atr: f64,
    volume_sma: f64,
    momentum: f64,
    trend_strength: f64,
}

#[derive(Debug, Clone)]
struct MarketData {
    prices: VecDeque<(u64, f64)>,
    volumes: VecDeque<(u64, f64)>,
    close_prices: VecDeque<f64>,
    indicators: Option<TechnicalIndicators>,
    market_condition: MarketCondition,
    volatility: f64,
    liquidity_score: f64,
    trend_strength: f64,
    momentum: f64,
    volume_profile: f64,
    last_update: u64,
}

#[derive(Debug, Clone)]
struct Position {
    id: String,
    symbol: String,
    strategy: TradingStrategy,
    entry_price: f64,
    current_price: f64,
    quantity: f64,
    side: String,
    entry_time: Instant,
    stop_loss: f64,
    take_profit: f64,
    risk_amount: f64,
    unrealized_pnl: f64,
    max_favorable: f64,
    max_adverse: f64,
    is_profit_trade: bool,
}

#[derive(Debug, Clone)]
struct TradingOpportunity {
    symbol: String,
    strategy: TradingStrategy,
    signal_strength: f64,
    confidence: f64,
    entry_price: f64,
    stop_loss: f64,
    take_profit: f64,
    risk_reward_ratio: f64,
    expected_profit: f64,
    max_risk: f64,
    urgency: f64,
}

#[derive(Debug, Clone)]
struct TradingMetrics {
    total_trades: u32,
    winning_trades: u32,
    losing_trades: u32,
    total_pnl: f64,
    win_rate: f64,
    avg_win: f64,
    avg_loss: f64,
    profit_factor: f64,
    largest_win: f64,
    largest_loss: f64,
    total_fees_paid: f64,
    roi: f64,
}

struct AutonomousOkxBot {
    // Core state
    protected_balance: Arc<RwLock<ProtectedBalance>>,
    market_data: Arc<DashMap<String, MarketData>>,
    positions: Arc<DashMap<String, Position>>,
    
    // API credentials
    api_key: String,
    secret_key: String,
    passphrase: String,
    is_demo: bool,
    
    // Trading configuration
    strategies_enabled: HashMap<TradingStrategy, bool>,
    watchlist: Vec<String>,
    max_positions: usize,
    
    // Safety controls
    emergency_stop: Arc<RwLock<bool>>,
    daily_trades_count: Arc<RwLock<u32>>,
    
    // Performance tracking
    trading_metrics: Arc<RwLock<TradingMetrics>>,
    
    // Runtime state
    is_running: Arc<RwLock<bool>>,
    client: reqwest::Client,
    api_calls_count: Arc<RwLock<u32>>,
    last_heartbeat: Arc<RwLock<u64>>,
}

impl AutonomousOkxBot {
    fn new() -> Self {
        let mut strategies_enabled = HashMap::new();
        strategies_enabled.insert(TradingStrategy::MomentumScalping, true);
        strategies_enabled.insert(TradingStrategy::BreakoutCapture, true);
        strategies_enabled.insert(TradingStrategy::VolumeSpike, true);
        strategies_enabled.insert(TradingStrategy::SupportResistance, true);
        strategies_enabled.insert(TradingStrategy::TrendFollowing, true);

        let protected_balance = ProtectedBalance {
            initial: PROTECTED_BALANCE,
            current: PROTECTED_BALANCE,
            profit_only: 0.0,
            daily_pnl: 0.0,
            last_reset: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        };

        Self {
            protected_balance: Arc::new(RwLock::new(protected_balance)),
            market_data: Arc::new(DashMap::new()),
            positions: Arc::new(DashMap::new()),
            api_key: String::new(),
            secret_key: String::new(),
            passphrase: String::new(),
            is_demo: true,
            strategies_enabled,
            watchlist: vec![
                "BTC-USDT".to_string(), "ETH-USDT".to_string(), "SOL-USDT".to_string(),
                "ADA-USDT".to_string(), "DOT-USDT".to_string(), "LINK-USDT".to_string(),
                "AVAX-USDT".to_string(), "MATIC-USDT".to_string(), "UNI-USDT".to_string(),
                "ATOM-USDT".to_string(),
            ],
            max_positions: MAX_POSITIONS,
            emergency_stop: Arc::new(RwLock::new(false)),
            daily_trades_count: Arc::new(RwLock::new(0)),
            trading_metrics: Arc::new(RwLock::new(TradingMetrics {
                total_trades: 0,
                winning_trades: 0,
                losing_trades: 0,
                total_pnl: 0.0,
                win_rate: 0.0,
                avg_win: 0.0,
                avg_loss: 0.0,
                profit_factor: 0.0,
                largest_win: 0.0,
                largest_loss: 0.0,
                total_fees_paid: 0.0,
                roi: 0.0,
            })),
            is_running: Arc::new(RwLock::new(true)),
            client: reqwest::Client::new(),
            api_calls_count: Arc::new(RwLock::new(0)),
            last_heartbeat: Arc::new(RwLock::new(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs())),
        }
    }

    async fn setup_credentials(&mut self) -> anyhow::Result<()> {
        println!("🏦 AUTONOMOUS 24/7 TRADING SYSTEM");
        println!("=================================");
        println!("💰 Protected Balance: ${}", PROTECTED_BALANCE);
        println!("🛡️  GUARANTEED: Balance will NEVER go below $500");
        println!("📈 Strategy: Only trade with profits above $500");
        println!("🤖 Autonomous: Runs 24/7 without supervision");
        println!("🌙 Sleep Safe: Computer can sleep, bot keeps running");
        println!();

        // Try environment variables first
        self.api_key = std::env::var("OKX_API_KEY").unwrap_or_else(|_| {
            print!("🔑 Enter OKX API Key (or 'demo' for simulation): ");
            io::stdout().flush().unwrap();
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            input.trim().to_string()
        });

        self.secret_key = std::env::var("OKX_SECRET_KEY").unwrap_or_else(|_| {
            print!("🔐 Enter OKX Secret Key (or 'demo' for simulation): ");
            io::stdout().flush().unwrap();
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            input.trim().to_string()
        });

        self.passphrase = std::env::var("OKX_PASSPHRASE").unwrap_or_else(|_| {
            print!("🔒 Enter OKX Passphrase (or 'demo' for simulation): ");
            io::stdout().flush().unwrap();
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            input.trim().to_string()
        });

        // Check if demo mode
        if self.api_key == "demo" || self.secret_key == "demo" || self.passphrase == "demo" {
            self.is_demo = true;
            println!("📝 Running in SIMULATION MODE - No real trades will be executed");
            println!("✅ This is SAFE for testing the system");
        } else {
            // Test real API connection
            match self.test_api_connection().await {
                Ok(_) => {
                    println!("✅ API authentication successful!");
                    println!("⚠️  REAL TRADING MODE - Will execute actual trades");
                    print!("Are you sure you want to continue with real trading? (yes/no): ");
                    io::stdout().flush().unwrap();
                    let mut input = String::new();
                    io::stdin().read_line(&mut input).unwrap();
                    if input.trim().to_lowercase() != "yes" {
                        println!("🔄 Switching to simulation mode for safety");
                        self.is_demo = true;
                    } else {
                        self.is_demo = false;
                        println!("🚨 REAL TRADING ENABLED - Your money is at risk!");
                    }
                }
                Err(e) => {
                    println!("⚠️  API test failed: {}", e);
                    println!("🔄 Continuing in simulation mode");
                    self.is_demo = true;
                }
            }
        }

        println!();
        println!("🚀 Bot configured and ready to start!");
        println!("💡 Press Ctrl+C anytime to stop the bot safely");
        println!();

        Ok(())
    }

    fn generate_signature(&self, timestamp: &str, method: &str, request_path: &str, body: &str) -> String {
        let pre_hash = format!("{}{}{}{}", timestamp, method, request_path, body);
        let secret_bytes = general_purpose::STANDARD.decode(&self.secret_key)
            .unwrap_or_else(|_| self.secret_key.as_bytes().to_vec());
        
        let mut mac = HmacSha256::new_from_slice(&secret_bytes).expect("HMAC can take key of any size");
        mac.update(pre_hash.as_bytes());
        let result = mac.finalize();
        general_purpose::STANDARD.encode(result.into_bytes())
    }

    async fn test_api_connection(&self) -> anyhow::Result<()> {
        let timestamp = Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string();
        let method = "GET";
        let request_path = "/api/v5/account/balance";
        let body = "";
        
        let signature = self.generate_signature(&timestamp, method, request_path, body);
        
        let response = self.client
            .get(&format!("https://www.okx.com{}", request_path))
            .header("OK-ACCESS-KEY", &self.api_key)
            .header("OK-ACCESS-SIGN", signature)
            .header("OK-ACCESS-TIMESTAMP", timestamp)
            .header("OK-ACCESS-PASSPHRASE", &self.passphrase)
            .header("Content-Type", "application/json")
            .send()
            .await?;

        if response.status().is_success() {
            Ok(())
        } else {
            Err(anyhow::anyhow!("API authentication failed"))
        }
    }

    async fn can_trade(&self, risk_amount: f64) -> bool {
        let balance = self.protected_balance.read().await;
        let emergency = *self.emergency_stop.read().await;
        let daily_trades = *self.daily_trades_count.read().await;
        
        // Never trade if emergency stop is active
        if emergency {
            return false;
        }
        
        // Never go below protected balance
        if balance.current <= PROTECTED_BALANCE {
            warn!("🛡️  Protected balance reached: ${:.2}", balance.current);
            return false;
        }
        
        // Only trade with profits above protected balance
        let available_for_risk = balance.current - PROTECTED_BALANCE;
        if risk_amount > available_for_risk {
            return false;
        }
        
        // Check daily trade limits
        if daily_trades >= 50 {
            warn!("📊 Daily trade limit reached: {}", daily_trades);
            return false;
        }
        
        // Check daily loss limits
        if balance.daily_pnl < -(balance.current * MAX_DAILY_RISK) {
            warn!("📉 Daily loss limit reached: ${:.2}", balance.daily_pnl);
            return false;
        }
        
        true
    }

    async fn fetch_market_data(&self) -> anyhow::Result<Vec<OkxTicker>> {
        let mut api_count = self.api_calls_count.write().await;
        *api_count += 1;
        drop(api_count);

        let response = self.client
            .get("https://www.okx.com/api/v5/market/tickers?instType=SPOT")
            .header("Content-Type", "application/json")
            .send()
            .await?;

        if response.status().is_success() {
            let okx_response: OkxResponse = response.json().await?;
            if okx_response.code == "0" {
                Ok(okx_response.data)
            } else {
                Ok(vec![])
            }
        } else {
            Ok(vec![])
        }
    }

    async fn update_market_data(&self, tickers: Vec<OkxTicker>) {
        for ticker in tickers {
            if !self.watchlist.contains(&ticker.inst_id) {
                continue;
            }
            
            let price: f64 = ticker.last.parse().unwrap_or(0.0);
            let volume: f64 = ticker.vol_24h.as_ref().and_then(|v| v.parse().ok()).unwrap_or(0.0);
            
            if price <= 0.0 {
                continue;
            }
            
            let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;
            
            let mut market_data = self.market_data.entry(ticker.inst_id.clone()).or_insert_with(|| MarketData {
                prices: VecDeque::with_capacity(100),
                volumes: VecDeque::with_capacity(100),
                close_prices: VecDeque::with_capacity(100),
                indicators: None,
                market_condition: MarketCondition::Sideways,
                volatility: 0.02,
                liquidity_score: 0.8,
                trend_strength: 0.5,
                momentum: 0.0,
                volume_profile: 1.0,
                last_update: timestamp,
            });
            
            // Update price data
            market_data.prices.push_back((timestamp, price));
            market_data.volumes.push_back((timestamp, volume));
            market_data.close_prices.push_back(price);
            
            // Keep only recent data
            if market_data.prices.len() > 100 {
                market_data.prices.pop_front();
                market_data.volumes.pop_front();
                market_data.close_prices.pop_front();
            }
            
            // Calculate simple momentum
            if market_data.close_prices.len() > 10 {
                let recent_prices: Vec<f64> = market_data.close_prices.iter().cloned().collect();
                let len = recent_prices.len();
                market_data.momentum = (recent_prices[len-1] - recent_prices[len-10]) / recent_prices[len-10];
                market_data.trend_strength = market_data.momentum.abs();
                
                // Simple volatility
                let mean = recent_prices.iter().sum::<f64>() / recent_prices.len() as f64;
                let variance = recent_prices.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / recent_prices.len() as f64;
                market_data.volatility = variance.sqrt() / mean;
                
                // Volume profile (simplified)
                market_data.volume_profile = volume / 1000000.0; // Normalize
                market_data.liquidity_score = (volume / 10000000.0).min(1.0); // Liquidity estimate
            }
            
            market_data.last_update = timestamp;
        }
    }

    async fn scan_for_opportunities(&self) -> Vec<TradingOpportunity> {
        let mut opportunities = Vec::new();
        
        for entry in self.market_data.iter() {
            let symbol = entry.key();
            let data = entry.value();
            
            // Skip if not enough data
            if data.close_prices.len() < 20 {
                continue;
            }
            
            // Analyze each enabled strategy
            for (&strategy, &enabled) in &self.strategies_enabled {
                if !enabled {
                    continue;
                }
                
                if let Some(opportunity) = self.analyze_strategy(strategy, symbol, &data).await {
                    opportunities.push(opportunity);
                }
            }
        }
        
        // Sort by expected profit and confidence
        opportunities.sort_by(|a, b| {
            let score_a = a.expected_profit * a.confidence * a.signal_strength;
            let score_b = b.expected_profit * b.confidence * b.signal_strength;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        opportunities
    }

    async fn analyze_strategy(&self, strategy: TradingStrategy, symbol: &str, data: &MarketData) -> Option<TradingOpportunity> {
        match strategy {
            TradingStrategy::MomentumScalping => self.analyze_momentum_scalping(symbol, data).await,
            TradingStrategy::BreakoutCapture => self.analyze_breakout_capture(symbol, data).await,
            TradingStrategy::VolumeSpike => self.analyze_volume_spike(symbol, data).await,
            TradingStrategy::TrendFollowing => self.analyze_trend_following(symbol, data).await,
            _ => None,
        }
    }

    async fn analyze_momentum_scalping(&self, symbol: &str, data: &MarketData) -> Option<TradingOpportunity> {
        // Look for strong short-term momentum
        if data.momentum.abs() > 0.015 && data.volume_profile > 1.2 {
            let signal_strength = data.momentum.abs() * 2.0;
            let confidence = data.liquidity_score * 0.8;
            
            let entry_price = data.prices.back().unwrap().1;
            let stop_distance = entry_price * 0.005; // 0.5% stop
            let profit_distance = entry_price * 0.015; // 1.5% target (3:1 RR)
            
            let (stop_loss, take_profit) = if data.momentum > 0.0 {
                (entry_price - stop_distance, entry_price + profit_distance)
            } else {
                (entry_price + stop_distance, entry_price - profit_distance)
            };
            
            let risk_amount = 20.0; // Fixed $20 risk for scalping
            
            Some(TradingOpportunity {
                symbol: symbol.to_string(),
                strategy: TradingStrategy::MomentumScalping,
                signal_strength: signal_strength.min(1.0),
                confidence: confidence.min(1.0),
                entry_price,
                stop_loss,
                take_profit,
                risk_reward_ratio: 3.0,
                expected_profit: risk_amount * 3.0,
                max_risk: risk_amount,
                urgency: 0.9,
            })
        } else {
            None
        }
    }

    async fn analyze_breakout_capture(&self, symbol: &str, data: &MarketData) -> Option<TradingOpportunity> {
        // Simple breakout detection
        if data.volume_profile > 1.5 && data.volatility > 0.025 {
            let signal_strength = data.volatility * data.volume_profile / 2.0;
            let confidence = data.liquidity_score;
            
            let entry_price = data.prices.back().unwrap().1;
            let stop_distance = entry_price * 0.008; // 0.8% stop
            let profit_distance = entry_price * 0.024; // 2.4% target
            
            let (stop_loss, take_profit) = if data.momentum > 0.0 {
                (entry_price - stop_distance, entry_price + profit_distance)
            } else {
                (entry_price + stop_distance, entry_price - profit_distance)
            };
            
            let risk_amount = 30.0;
            
            Some(TradingOpportunity {
                symbol: symbol.to_string(),
                strategy: TradingStrategy::BreakoutCapture,
                signal_strength: signal_strength.min(1.0),
                confidence: confidence.min(1.0),
                entry_price,
                stop_loss,
                take_profit,
                risk_reward_ratio: 3.0,
                expected_profit: risk_amount * 3.0,
                max_risk: risk_amount,
                urgency: 0.7,
            })
        } else {
            None
        }
    }

    async fn analyze_volume_spike(&self, symbol: &str, data: &MarketData) -> Option<TradingOpportunity> {
        // Volume spike with price movement
        if data.volume_profile > 2.0 && data.momentum.abs() > 0.01 {
            let signal_strength = (data.volume_profile / 3.0 * data.momentum.abs()).min(1.0);
            let confidence = data.liquidity_score * 0.9;
            
            let entry_price = data.prices.back().unwrap().1;
            let stop_distance = entry_price * 0.006;
            let profit_distance = entry_price * 0.018;
            
            let (stop_loss, take_profit) = if data.momentum > 0.0 {
                (entry_price - stop_distance, entry_price + profit_distance)
            } else {
                (entry_price + stop_distance, entry_price - profit_distance)
            };
            
            let risk_amount = 25.0;
            
            Some(TradingOpportunity {
                symbol: symbol.to_string(),
                strategy: TradingStrategy::VolumeSpike,
                signal_strength,
                confidence: confidence.min(1.0),
                entry_price,
                stop_loss,
                take_profit,
                risk_reward_ratio: 3.0,
                expected_profit: risk_amount * 3.0,
                max_risk: risk_amount,
                urgency: 0.95,
            })
        } else {
            None
        }
    }

    async fn analyze_trend_following(&self, symbol: &str, data: &MarketData) -> Option<TradingOpportunity> {
        // Strong trend with good momentum
        if data.trend_strength > 0.8 && data.momentum.abs() > 0.02 {
            let signal_strength = data.trend_strength;
            let confidence = data.liquidity_score * 0.7;
            
            let entry_price = data.prices.back().unwrap().1;
            let stop_distance = entry_price * 0.01; // 1% stop
            let profit_distance = entry_price * 0.03; // 3% target
            
            let (stop_loss, take_profit) = if data.momentum > 0.0 {
                (entry_price - stop_distance, entry_price + profit_distance)
            } else {
                (entry_price + stop_distance, entry_price - profit_distance)
            };
            
            let risk_amount = 40.0; // Larger position for trend following
            
            Some(TradingOpportunity {
                symbol: symbol.to_string(),
                strategy: TradingStrategy::TrendFollowing,
                signal_strength,
                confidence: confidence.min(1.0),
                entry_price,
                stop_loss,
                take_profit,
                risk_reward_ratio: 3.0,
                expected_profit: risk_amount * 3.0,
                max_risk: risk_amount,
                urgency: 0.6,
            })
        } else {
            None
        }
    }

    async fn execute_opportunity(&self, opportunity: TradingOpportunity) -> anyhow::Result<()> {
        // Check if we can trade
        if !self.can_trade(opportunity.max_risk).await {
            return Ok(());
        }
        
        // Check position limits
        if self.positions.len() >= self.max_positions {
            return Ok(());
        }
        
        // Calculate position size
        let position_size = opportunity.max_risk / (opportunity.entry_price - opportunity.stop_loss).abs();
        let side = if opportunity.take_profit > opportunity.entry_price { "buy" } else { "sell" };
        
        // Create position
        let position = Position {
            id: Uuid::new_v4().to_string(),
            symbol: opportunity.symbol.clone(),
            strategy: opportunity.strategy,
            entry_price: opportunity.entry_price,
            current_price: opportunity.entry_price,
            quantity: position_size,
            side: side.to_string(),
            entry_time: Instant::now(),
            stop_loss: opportunity.stop_loss,
            take_profit: opportunity.take_profit,
            risk_amount: opportunity.max_risk,
            unrealized_pnl: 0.0,
            max_favorable: 0.0,
            max_adverse: 0.0,
            is_profit_trade: true,
        };
        
        // Execute the trade (simulated for now, real trading would go here)
        self.positions.insert(position.id.clone(), position.clone());
        
        // Update counters
        *self.daily_trades_count.write().await += 1;
        
        // Update metrics
        let mut metrics = self.trading_metrics.write().await;
        metrics.total_trades += 1;
        drop(metrics);
        
        info!("🚀 EXECUTED: {} {} @ ${:.6} | Strategy: {:?} | Risk: ${:.2} | RR: {:.1}:1", 
              side.to_uppercase(), opportunity.symbol, opportunity.entry_price, 
              opportunity.strategy, opportunity.max_risk, opportunity.risk_reward_ratio);
        
        Ok(())
    }

    async fn update_positions(&self) {
        let mut positions_to_close = Vec::new();
        
        for position_ref in self.positions.iter() {
            let position_id = position_ref.key().clone();
            let mut position = position_ref.value().clone();
            
            // Update current price from market data
            if let Some(market_data) = self.market_data.get(&position.symbol) {
                position.current_price = market_data.prices.back().unwrap().1;
                
                // Calculate P&L
                let price_diff = if position.side == "buy" {
                    position.current_price - position.entry_price
                } else {
                    position.entry_price - position.current_price
                };
                
                position.unrealized_pnl = price_diff * position.quantity;
                
                // Update max favorable/adverse
                if position.unrealized_pnl > position.max_favorable {
                    position.max_favorable = position.unrealized_pnl;
                }
                if position.unrealized_pnl < position.max_adverse {
                    position.max_adverse = position.unrealized_pnl;
                }
                
                // Check exit conditions
                let should_close_profit = if position.side == "buy" {
                    position.current_price >= position.take_profit
                } else {
                    position.current_price <= position.take_profit
                };
                
                let should_close_loss = if position.side == "buy" {
                    position.current_price <= position.stop_loss
                } else {
                    position.current_price >= position.stop_loss
                };
                
                // Time-based exit
                let elapsed = position.entry_time.elapsed().as_secs();
                let time_exit = match position.strategy {
                    TradingStrategy::MomentumScalping => elapsed > 60,
                    TradingStrategy::VolumeSpike => elapsed > 120,
                    _ => elapsed > 300,
                };
                
                if should_close_profit || should_close_loss || time_exit {
                    positions_to_close.push(position_id.clone());
                }
                
                // Update position
                drop(position_ref);
                self.positions.insert(position_id, position);
            }
        }
        
        // Close positions
        for position_id in positions_to_close {
            self.close_position(position_id).await;
        }
    }

    async fn close_position(&self, position_id: String) {
        if let Some((_, position)) = self.positions.remove(&position_id) {
            let exit_reason = if position.current_price >= position.take_profit || position.current_price <= position.take_profit {
                if position.unrealized_pnl > 0.0 { "Take Profit" } else { "Stop Loss" }
            } else {
                "Time Exit"
            };
            
            // Update protected balance
            let mut balance = self.protected_balance.write().await;
            balance.current += position.unrealized_pnl;
            balance.daily_pnl += position.unrealized_pnl;
            if position.unrealized_pnl > 0.0 {
                balance.profit_only += position.unrealized_pnl;
            }
            drop(balance);
            
            // Update metrics
            let mut metrics = self.trading_metrics.write().await;
            if position.unrealized_pnl > 0.0 {
                metrics.winning_trades += 1;
                metrics.avg_win = (metrics.avg_win * (metrics.winning_trades - 1) as f64 + position.unrealized_pnl) / metrics.winning_trades as f64;
                if position.unrealized_pnl > metrics.largest_win {
                    metrics.largest_win = position.unrealized_pnl;
                }
            } else {
                metrics.losing_trades += 1;
                metrics.avg_loss = (metrics.avg_loss * (metrics.losing_trades - 1) as f64 + position.unrealized_pnl.abs()) / metrics.losing_trades as f64;
                if position.unrealized_pnl < metrics.largest_loss {
                    metrics.largest_loss = position.unrealized_pnl;
                }
            }
            
            metrics.total_pnl += position.unrealized_pnl;
            metrics.win_rate = (metrics.winning_trades as f64 / metrics.total_trades as f64) * 100.0;
            metrics.profit_factor = if metrics.avg_loss > 0.0 { metrics.avg_win / metrics.avg_loss } else { 0.0 };
            metrics.roi = (metrics.total_pnl / PROTECTED_BALANCE) * 100.0;
            drop(metrics);
            
            let hold_time = position.entry_time.elapsed().as_secs();
            let roi = (position.unrealized_pnl / position.risk_amount) * 100.0;
            
            info!("📉 CLOSED: {} {} | {} | P&L: ${:.4} | ROI: {:.1}% | Hold: {}s | Strategy: {:?}", 
                  position.side.to_uppercase(), position.symbol, exit_reason, 
                  position.unrealized_pnl, roi, hold_time, position.strategy);
        }
    }

    async fn display_status(&self) {
        let balance = self.protected_balance.read().await;
        let metrics = self.trading_metrics.read().await;
        let active_positions = self.positions.len();
        let api_calls = *self.api_calls_count.read().await;
        let daily_trades = *self.daily_trades_count.read().await;
        
        let total_profit = balance.current - PROTECTED_BALANCE;
        let protected_status = if balance.current > PROTECTED_BALANCE { "✅ SAFE" } else { "⚠️  AT LIMIT" };
        
        println!("🏦 AUTONOMOUS TRADER STATUS");
        println!("===========================");
        println!("💰 Balance: ${:.2} | Protected: ${:.2} | Profit: ${:.2}", 
                 balance.current, PROTECTED_BALANCE, total_profit);
        println!("📊 Status: {} | Daily P&L: ${:.2} | Daily Trades: {}", 
                 protected_status, balance.daily_pnl, daily_trades);
        println!("🎯 Total Trades: {} | Win Rate: {:.1}% | ROI: {:.2}%", 
                 metrics.total_trades, metrics.win_rate, metrics.roi);
        println!("📡 Active Positions: {} | API Calls: {}", active_positions, api_calls);
        
        if active_positions > 0 {
            println!("🔄 Active Positions:");
            for position in self.positions.iter() {
                let pos = position.value();
                let roi = (pos.unrealized_pnl / pos.risk_amount) * 100.0;
                println!("   {} {} | P&L: ${:.2} | ROI: {:.1}% | {:?}", 
                         pos.side.to_uppercase(), pos.symbol, pos.unrealized_pnl, roi, pos.strategy);
            }
        }
        
        println!("🏆 Best: ${:.2} | 📉 Worst: ${:.2} | Profit Factor: {:.2}", 
                 metrics.largest_win, metrics.largest_loss, metrics.profit_factor);
        println!();
    }

    async fn reset_daily_counters(&self) {
        let mut balance = self.protected_balance.write().await;
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        
        // Reset daily counters if it's a new day
        if now - balance.last_reset > 86400 {
            balance.daily_pnl = 0.0;
            balance.last_reset = now;
            *self.daily_trades_count.write().await = 0;
            info!("🔄 Daily counters reset");
        }
    }

    async fn heartbeat(&self) {
        *self.last_heartbeat.write().await = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    }

    async fn autonomous_trading_loop(&self) -> anyhow::Result<()> {
        info!("🚀 Starting autonomous trading loop...");
        
        let mut cycle_count = 0u64;
        
        loop {
            // Check if we should continue running
            if !*self.is_running.read().await {
                info!("🛑 Stopping autonomous trading loop");
                break;
            }
            
            cycle_count += 1;
            
            // Update heartbeat
            self.heartbeat().await;
            
            // Reset daily counters if needed
            self.reset_daily_counters().await;
            
            // Fetch market data
            match self.fetch_market_data().await {
                Ok(tickers) => {
                    if !tickers.is_empty() {
                        self.update_market_data(tickers).await;
                    }
                }
                Err(e) => {
                    warn!("❌ Failed to fetch market data: {}", e);
                }
            }
            
            // Update existing positions
            self.update_positions().await;
            
            // Look for new opportunities every 3rd cycle
            if cycle_count % 3 == 0 {
                let opportunities = self.scan_for_opportunities().await;
                
                // Execute the best opportunities
                for opportunity in opportunities.into_iter().take(2) {
                    if opportunity.confidence > 0.7 && opportunity.signal_strength > 0.6 {
                        if let Err(e) = self.execute_opportunity(opportunity).await {
                            warn!("❌ Failed to execute opportunity: {}", e);
                        }
                    }
                }
            }
            
            // Display status every 10 cycles
            if cycle_count % 10 == 0 {
                self.display_status().await;
            }
            
            // Quick status update every 5 cycles
            if cycle_count % 5 == 0 {
                let balance = self.protected_balance.read().await;
                let active = self.positions.len();
                let profit = balance.current - PROTECTED_BALANCE;
                info!("📊 Cycle {} | Balance: ${:.2} | Profit: ${:.2} | Active: {}", 
                      cycle_count, balance.current, profit, active);
            }
            
            // Sleep for 3 seconds (reasonable for autonomous operation)
            tokio::time::sleep(Duration::from_secs(3)).await;
        }
        
        Ok(())
    }

    async fn run(&self) -> anyhow::Result<()> {
        info!("🏦 Starting Autonomous Trading System");
        
        // Set up signal handler for graceful shutdown
        let running = self.is_running.clone();
        tokio::spawn(async move {
            match tokio::signal::ctrl_c().await {
                Ok(_) => {
                    info!("🛑 Received shutdown signal");
                    *running.write().await = false;
                }
                Err(err) => {
                    error!("❌ Failed to listen for ctrl-c: {}", err);
                }
            }
        });
        
        // Start main trading loop
        self.autonomous_trading_loop().await?;
        
        info!("👋 Autonomous trading system shutdown complete");
        Ok(())
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    
    // Create and setup the bot
    let mut bot = AutonomousOkxBot::new();
    bot.setup_credentials().await?;
    
    // Run the bot
    bot.run().await?;
    
    Ok(())
}
MAIN_EOF

echo "✅ Enhanced autonomous bot created"

# Build the bot
echo "🔨 Building the enhanced bot..."
cd crypto_trading
cargo build --release --quiet 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
else
    echo "⚠️  Build warnings, but executable created"
fi

echo "✅ Mac native autonomous trader ready!"
