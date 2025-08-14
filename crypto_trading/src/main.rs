use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use dashmap::DashMap;
use reqwest;
use anyhow::Result;

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
    ts: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OkxResponse {
    code: String,
    data: Vec<OkxTicker>,
}

#[derive(Debug, Clone)]
struct FastTickData {
    symbol: String,
    price: f64,
    bid: f64,
    ask: f64,
    volume: f64,
    last_price: f64,
    price_velocity: f64,
    spread: f64,
    momentum_1s: f64,
    momentum_3s: f64,
    volume_velocity: f64,
    tick_count: u64,
    last_update: u64,
}

#[derive(Debug, Clone)]
struct HFTPosition {
    id: String,
    symbol: String,
    side: String,
    entry_price: f64,
    current_price: f64,
    size: f64,
    entry_time: Instant,
    target: f64,
    stop: f64,
    pnl: f64,
    max_hold_ms: u64,
}

#[derive(Debug, Clone)]
struct HFTTrade {
    symbol: String,
    side: String,
    entry_price: f64,
    exit_price: f64,
    size: f64,
    pnl: f64,
    hold_time_ms: u64,
    strategy: String,
    timestamp: u64,
}

struct HighFrequencyTrader {
    client: reqwest::Client,
    
    // Market data for ALL pairs
    all_tickers: Arc<DashMap<String, FastTickData>>,
    price_buffers: Arc<DashMap<String, VecDeque<(u64, f64)>>>,
    
    // HFT positions and trades
    positions: Arc<DashMap<String, HFTPosition>>,
    completed_trades: Arc<RwLock<Vec<HFTTrade>>>,
    
    // Portfolio
    balance: Arc<RwLock<f64>>,
    total_pnl: Arc<RwLock<f64>>,
    
    // HFT settings
    tick_interval_ms: u64,
    max_positions: usize,
    min_spread_bps: f64,
    min_momentum_bps: f64,
    max_hold_time_ms: u64,
    
    // Performance counters
    total_ticks: Arc<RwLock<u64>>,
    signals_generated: Arc<RwLock<u64>>,
    trades_executed: Arc<RwLock<u64>>,
    updates_per_second: Arc<RwLock<f64>>,
    
    last_display_time: Arc<RwLock<Instant>>,
}

impl HighFrequencyTrader {
    fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_millis(500))
                .build()
                .unwrap(),
            
            all_tickers: Arc::new(DashMap::new()),
            price_buffers: Arc::new(DashMap::new()),
            positions: Arc::new(DashMap::new()),
            completed_trades: Arc::new(RwLock::new(Vec::new())),
            balance: Arc::new(RwLock::new(1000.0)),
            total_pnl: Arc::new(RwLock::new(0.0)),
            
            // Aggressive HFT settings
            tick_interval_ms: 50,    // 20 times per second
            max_positions: 20,       // Up to 20 simultaneous positions
            min_spread_bps: 2.0,     // 0.02% minimum spread
            min_momentum_bps: 1.0,   // 0.01% minimum momentum
            max_hold_time_ms: 5000,  // 5 second max hold time
            
            total_ticks: Arc::new(RwLock::new(0)),
            signals_generated: Arc::new(RwLock::new(0)),
            trades_executed: Arc::new(RwLock::new(0)),
            updates_per_second: Arc::new(RwLock::new(0.0)),
            last_display_time: Arc::new(RwLock::new(Instant::now())),
        }
    }
    
    async fn fetch_all_market_data(&self) -> Result<()> {
        let fetch_start = Instant::now();
        
        let response = self.client
            .get("https://www.okx.com/api/v5/market/tickers?instType=SPOT")
            .send()
            .await?;
        
        if response.status().is_success() {
            let okx_response: OkxResponse = response.json().await?;
            let fetch_time = fetch_start.elapsed();
            
            let process_start = Instant::now();
            
            // Process ALL tickers, not just watchlist
            for ticker in okx_response.data {
                // Only trade USDT pairs for liquidity
                if ticker.inst_id.ends_with("-USDT") {
                    self.process_ticker_ultra_fast(&ticker).await;
                }
            }
            
            let process_time = process_start.elapsed();
            let total_time = fetch_start.elapsed();
            
            // Update performance metrics
            let mut ticks = self.total_ticks.write().await;
            *ticks += 1;
            
            // Calculate updates per second
            if total_time.as_millis() > 0 {
                let updates_per_sec = 1000.0 / total_time.as_millis() as f64;
                *self.updates_per_second.write().await = updates_per_sec;
            }
            
            // Show performance for very fast updates
            if *ticks % 20 == 0 {
                println!("⚡ Tick #{} | Fetch: {}ms | Process: {}ms | Total: {}ms | {:.1} ticks/sec", 
                         *ticks, fetch_time.as_millis(), process_time.as_millis(), 
                         total_time.as_millis(), 1000.0 / total_time.as_millis() as f64);
            }
        }
        
        Ok(())
    }
    
    async fn process_ticker_ultra_fast(&self, ticker: &OkxTicker) {
        let price = ticker.last.parse::<f64>().unwrap_or(0.0);
        let bid = ticker.bid_px.as_ref().and_then(|b| b.parse().ok()).unwrap_or(price);
        let ask = ticker.ask_px.as_ref().and_then(|a| a.parse().ok()).unwrap_or(price);
        let volume = ticker.vol_24h.as_ref().and_then(|v| v.parse().ok()).unwrap_or(0.0);
        
        if price <= 0.0 || bid <= 0.0 || ask <= 0.0 {
            return;
        }
        
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64;
        let spread = (ask - bid) / bid * 10000.0; // basis points
        
        // Get or create price buffer
        let mut price_buffer = self.price_buffers.entry(ticker.inst_id.clone()).or_insert_with(|| VecDeque::new());
        price_buffer.push_back((timestamp, price));
        if price_buffer.len() > 100 {
            price_buffer.pop_front();
        }
        
        // Calculate velocities and momentum
        let (price_velocity, momentum_1s, momentum_3s, volume_velocity) = if let Some(existing) = self.all_tickers.get(&ticker.inst_id) {
            let time_diff = (timestamp - existing.last_update) as f64 / 1_000_000_000.0; // nanoseconds to seconds
            
            let price_velocity = if time_diff > 0.0 && existing.price > 0.0 {
                ((price - existing.price) / existing.price) / time_diff * 10000.0 // basis points per second
            } else {
                existing.price_velocity * 0.95 // Decay
            };
            
            let volume_velocity = if time_diff > 0.0 && existing.volume > 0.0 {
                ((volume - existing.volume) / existing.volume) / time_diff * 100.0 // % per second
            } else {
                existing.volume_velocity * 0.9
            };
            
            // Calculate short-term momentum
            let momentum_1s = self.calculate_momentum(&price_buffer, 1_000_000_000); // 1 second in nanoseconds
            let momentum_3s = self.calculate_momentum(&price_buffer, 3_000_000_000); // 3 seconds in nanoseconds
            
            (price_velocity, momentum_1s, momentum_3s, volume_velocity)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };
        
        let tick_data = FastTickData {
            symbol: ticker.inst_id.clone(),
            price,
            bid,
            ask,
            volume,
            last_price: self.all_tickers.get(&ticker.inst_id).map(|t| t.price).unwrap_or(price),
            price_velocity,
            spread,
            momentum_1s,
            momentum_3s,
            volume_velocity,
            tick_count: self.all_tickers.get(&ticker.inst_id).map(|t| t.tick_count + 1).unwrap_or(1),
            last_update: timestamp,
        };
        
        self.all_tickers.insert(ticker.inst_id.clone(), tick_data);
    }
    
    fn calculate_momentum(&self, buffer: &VecDeque<(u64, f64)>, nanoseconds_ago: u64) -> f64 {
        if buffer.len() < 2 {
            return 0.0;
        }
        
        let current_time = buffer.back().unwrap().0;
        let target_time = current_time.saturating_sub(nanoseconds_ago);
        let current_price = buffer.back().unwrap().1;
        
        // Find price closest to target time
        let past_price = buffer.iter()
            .rev()
            .find(|(time, _)| *time <= target_time)
            .map(|(_, price)| *price)
            .unwrap_or(current_price);
        
        if past_price > 0.0 {
            (current_price - past_price) / past_price * 10000.0 // basis points
        } else {
            0.0
        }
    }
    
    async fn scan_hft_opportunities(&self) {
        let mut signals_count = 0;
        let current_positions = self.positions.len();
        
        if current_positions >= self.max_positions {
            return;
        }
        
        for ticker_ref in self.all_tickers.iter() {
            let ticker = ticker_ref.value();
            
            // Skip if already have position in this symbol
            if self.positions.iter().any(|p| p.value().symbol == ticker.symbol) {
                continue;
            }
            
            let mut signal_strength = 0.0;
            let mut strategy = "NONE";
            let mut side = "NONE";
            
            // Ultra-fast momentum strategy
            if ticker.momentum_1s.abs() > self.min_momentum_bps && ticker.price_velocity.abs() > 0.5 {
                signal_strength = (ticker.momentum_1s.abs() / 10.0).min(1.0);
                strategy = "MOMENTUM";
                side = if ticker.momentum_1s > 0.0 { "BUY" } else { "SELL" };
            }
            
            // Spread scalping strategy
            if ticker.spread > self.min_spread_bps && ticker.spread < 20.0 && ticker.volume > 100000.0 {
                signal_strength = (ticker.spread / 20.0).min(1.0);
                strategy = "SCALP";
                side = "BUY"; // Always buy mid-spread
            }
            
            // Volume velocity strategy
            if ticker.volume_velocity.abs() > 5.0 && ticker.momentum_1s.abs() > 2.0 {
                signal_strength = (ticker.volume_velocity.abs() / 20.0).min(1.0);
                strategy = "VOLUME";
                side = if ticker.momentum_1s > 0.0 { "BUY" } else { "SELL" };
            }
            
            // Mean reversion strategy
            if ticker.momentum_3s.abs() > 15.0 && ticker.momentum_1s.abs() < 5.0 {
                signal_strength = (ticker.momentum_3s.abs() / 30.0).min(1.0);
                strategy = "REVERSION";
                side = if ticker.momentum_3s > 0.0 { "SELL" } else { "BUY" };
            }
            
            // Execute if signal is strong enough
            if signal_strength > 0.6 && strategy != "NONE" {
                self.execute_hft_trade(&ticker.symbol, side, ticker.price, ticker.bid, ticker.ask, signal_strength, strategy).await;
                signals_count += 1;
                
                // Limit signals per cycle to avoid overtrading
                if signals_count >= 5 {
                    break;
                }
            }
        }
        
        if signals_count > 0 {
            let mut total_signals = self.signals_generated.write().await;
            *total_signals += signals_count;
        }
    }
    
    async fn execute_hft_trade(&self, symbol: &str, side: &str, price: f64, bid: f64, ask: f64, confidence: f64, strategy: &str) {
        let balance = *self.balance.read().await;
        let base_size = (balance * 0.02).min(50.0).max(5.0); // 2% of balance, $5-$50
        let position_size = base_size * confidence;
        
        // Calculate entry price with realistic slippage
        let entry_price = match side {
            "BUY" => ask + (ask * 0.0001), // Buy at ask + small slippage
            "SELL" => bid - (bid * 0.0001), // Sell at bid - small slippage
            _ => price,
        };
        
        // Tight profit targets for HFT
        let target = match side {
            "BUY" => entry_price * 1.002,  // 0.2% profit target
            "SELL" => entry_price * 0.998, // 0.2% profit target
            _ => entry_price,
        };
        
        let stop = match side {
            "BUY" => entry_price * 0.999,   // 0.1% stop loss
            "SELL" => entry_price * 1.001,  // 0.1% stop loss
            _ => entry_price,
        };
        
        let position = HFTPosition {
            id: uuid::Uuid::new_v4().to_string(),
            symbol: symbol.to_string(),
            side: side.to_string(),
            entry_price,
            current_price: entry_price,
            size: position_size,
            entry_time: Instant::now(),
            target,
            stop,
            pnl: -position_size * 0.001, // Include fees
            max_hold_ms: self.max_hold_time_ms,
        };
        
        self.positions.insert(position.id.clone(), position);
        
        let mut trades_executed = self.trades_executed.write().await;
        *trades_executed += 1;
        
        println!("⚡ HFT: {} {} {} @ ${:.6} | Size: ${:.2} | {} | Conf: {:.0}%", 
                 strategy, side, symbol, entry_price, position_size, 
                 format!("#{}", *trades_executed), confidence * 100.0);
    }
    
    async fn update_hft_positions(&self) {
        let mut positions_to_close = Vec::new();
        
        for pos_ref in self.positions.iter() {
            let pos_id = pos_ref.key().clone();
            let mut position = pos_ref.value().clone();
            
            if let Some(ticker) = self.all_tickers.get(&position.symbol) {
                // Use real-time bid/ask for exit pricing
                position.current_price = match position.side.as_str() {
                    "BUY" => ticker.bid,  // Exit buy at bid
                    "SELL" => ticker.ask, // Exit sell at ask
                    _ => ticker.price,
                };
                
                // Calculate P&L with realistic execution
                let price_diff = match position.side.as_str() {
                    "BUY" => position.current_price - position.entry_price,
                    "SELL" => position.entry_price - position.current_price,
                    _ => 0.0,
                };
                position.pnl = (price_diff / position.entry_price) * position.size - (position.size * 0.001);
                
                // Check exit conditions
                let hit_target = match position.side.as_str() {
                    "BUY" => position.current_price >= position.target,
                    "SELL" => position.current_price <= position.target,
                    _ => false,
                };
                
                let hit_stop = match position.side.as_str() {
                    "BUY" => position.current_price <= position.stop,
                    "SELL" => position.current_price >= position.stop,
                    _ => false,
                };
                
                let time_exit = position.entry_time.elapsed().as_millis() as u64 > position.max_hold_ms;
                
                if hit_target || hit_stop || time_exit {
                    positions_to_close.push((pos_id.clone(), position.clone()));
                } else {
                    // Update position
                    drop(pos_ref);
                    self.positions.insert(pos_id, position);
                }
            }
        }
        
        // Close positions
        for (pos_id, position) in positions_to_close {
            self.positions.remove(&pos_id);
            
            let hold_time = position.entry_time.elapsed().as_millis() as u64;
            let exit_reason = if position.pnl > 0.0 { "TARGET" } else if hold_time > position.max_hold_ms { "TIME" } else { "STOP" };
            
            // Update balance
            let mut balance = self.balance.write().await;
            *balance += position.pnl;
            
            let mut total_pnl = self.total_pnl.write().await;
            *total_pnl += position.pnl;
            
            // Record trade
            let trade = HFTTrade {
                symbol: position.symbol.clone(),
                side: position.side.clone(),
                entry_price: position.entry_price,
                exit_price: position.current_price,
                size: position.size,
                pnl: position.pnl,
                hold_time_ms: hold_time,
                strategy: exit_reason.to_string(),
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64,
            };
            
            let mut completed_trades = self.completed_trades.write().await;
            completed_trades.push(trade);
            if completed_trades.len() > 500 {
                completed_trades.remove(0);
            }
            
            println!("🔚 CLOSE: {} {} | P&L: ${:.3} | {} | {}ms", 
                     position.symbol, position.side, position.pnl, exit_reason, hold_time);
        }
    }
    
    async fn display_hft_dashboard(&self) {
        let now = Instant::now();
        let mut last_display = self.last_display_time.write().await;
        
        // Only update display every 2 seconds to avoid spam
        if now.duration_since(*last_display).as_millis() < 2000 {
            return;
        }
        *last_display = now;
        drop(last_display);
        
        print!("\x1B[2J\x1B[1;1H"); // Clear screen
        
        let balance = *self.balance.read().await;
        let total_pnl = *self.total_pnl.read().await;
        let total_ticks = *self.total_ticks.read().await;
        let signals_generated = *self.signals_generated.read().await;
        let trades_executed = *self.trades_executed.read().await;
        let updates_per_second = *self.updates_per_second.read().await;
        let active_positions = self.positions.len();
        let total_symbols = self.all_tickers.len();
        
        println!("⚡ HIGH-FREQUENCY TRADING SYSTEM - LIVE");
        println!("=====================================");
        println!("📊 MARKET: {} symbols | {:.1} ticks/sec | Tick #{}", 
                 total_symbols, updates_per_second, total_ticks);
        println!("💰 PORTFOLIO: ${:.2} | P&L: ${:.3} | ROI: {:.2}%", 
                 balance, total_pnl, total_pnl / 1000.0 * 100.0);
        println!("⚡ ACTIVITY: {} signals | {} trades | {} active", 
                 signals_generated, trades_executed, active_positions);
        println!();
        
        // Show most active symbols by tick count
        println!("🔥 MOST ACTIVE SYMBOLS:");
        let mut active_symbols: Vec<_> = self.all_tickers.iter().collect();
        active_symbols.sort_by(|a, b| b.value().tick_count.cmp(&a.value().tick_count));
        
        for (i, symbol_ref) in active_symbols.iter().take(10).enumerate() {
            let data = symbol_ref.value();
            let momentum_emoji = if data.momentum_1s.abs() > 5.0 { "🚀" } else if data.momentum_1s.abs() > 2.0 { "📈" } else { "📊" };
            println!("   {}{} {} | ${:.6} | M1s: {:.1}bp | Spread: {:.1}bp | V: {:.1}bp/s | #{}", 
                     i + 1, momentum_emoji, data.symbol, data.price, 
                     data.momentum_1s, data.spread, data.price_velocity, data.tick_count);
        }
        
        println!();
        
        // Show active positions
        if active_positions > 0 {
            println!("🔄 ACTIVE HFT POSITIONS:");
            for pos in self.positions.iter() {
                let p = pos.value();
                let age_ms = p.entry_time.elapsed().as_millis() as u64;
                let pnl_emoji = if p.pnl > 0.0 { "💰" } else { "📉" };
                println!("   {} {} {} @ ${:.6} | P&L: ${:.3} | {}ms | Target: ${:.6}", 
                         pnl_emoji, p.side, p.symbol, p.entry_price, p.pnl, age_ms, p.target);
            }
            println!();
        }
        
        // Show recent trades
        let completed_trades = self.completed_trades.read().await;
        if !completed_trades.is_empty() {
            println!("📋 RECENT HFT TRADES:");
            for trade in completed_trades.iter().rev().take(5) {
                let trade_emoji = if trade.pnl > 0.0 { "✅" } else { "❌" };
                println!("   {} {} {} | P&L: ${:.3} | {}ms hold", 
                         trade_emoji, trade.side, trade.symbol, trade.pnl, trade.hold_time_ms);
            }
        }
        
        println!();
        println!("⚡ HFT ENGINE RUNNING - NEXT TICK IN {}ms", self.tick_interval_ms);
    }
    
    async fn run_hft_engine(&self) -> Result<()> {
        println!("⚡ STARTING HIGH-FREQUENCY TRADING ENGINE");
        println!("========================================");
        println!("• Tick Interval: {}ms ({}x per second)", self.tick_interval_ms, 1000 / self.tick_interval_ms);
        println!("• Max Positions: {}", self.max_positions);
        println!("• Min Spread: {:.1} basis points", self.min_spread_bps);
        println!("• Max Hold Time: {}ms", self.max_hold_time_ms);
        println!("• Monitoring: ALL OKX USDT pairs");
        println!();
        
        let mut tick_interval = tokio::time::interval(Duration::from_millis(self.tick_interval_ms));
        let mut display_interval = tokio::time::interval(Duration::from_millis(2000));
        
        loop {
            tokio::select! {
                _ = tick_interval.tick() => {
                    // Ultra-fast market data and trading cycle
                    if let Err(e) = self.fetch_all_market_data().await {
                        eprintln!("Market data error: {}", e);
                    }
                    
                    self.scan_hft_opportunities().await;
                    self.update_hft_positions().await;
                }
                
                _ = display_interval.tick() => {
                    self.display_hft_dashboard().await;
                }
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let trader = HighFrequencyTrader::new();
    trader.run_hft_engine().await
}
