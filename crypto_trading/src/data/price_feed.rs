use std::collections::HashMap;
use std::time::Duration;
use tokio::time::interval;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use anyhow::Result;
use log::{info, error, debug};

use crate::core::types::MarketData;

pub struct WebSocketPriceFeed {
    url: String,
    subscriptions: Vec<String>,
    data_callback: Option<Box<dyn Fn(MarketData) + Send + Sync>>,
}

impl WebSocketPriceFeed {
    pub fn new(url: String) -> Self {
        Self {
            url,
            subscriptions: Vec::new(),
            data_callback: None,
        }
    }
    
    pub fn subscribe(&mut self, symbol: &str) {
        self.subscriptions.push(symbol.to_string());
    }
    
    pub fn set_callback<F>(&mut self, callback: F)
    where
        F: Fn(MarketData) + Send + Sync + 'static,
    {
        self.data_callback = Some(Box::new(callback));
    }
    
    pub async fn start(&self) -> Result<()> {
        let (ws_stream, _) = connect_async(&self.url).await?;
        let (mut write, mut read) = ws_stream.split();
        
        let subscribe_msg = json!({
            "op": "subscribe",
            "args": self.subscriptions.iter().map(|s| json!({
                "channel": "tickers",
                "instId": s
            })).collect::<Vec<_>>()
        });
        
        write.send(Message::Text(subscribe_msg.to_string())).await?;
        info!("Subscribed to {} symbols", self.subscriptions.len());
        
        while let Some(msg) = read.next().await {
            match msg? {
                Message::Text(text) => {
                    if let Ok(data) = serde_json::from_str::<Value>(&text) {
                        if let Some(ticker_data) = self.parse_ticker_message(&data) {
                            if let Some(ref callback) = self.data_callback {
                                callback(ticker_data);
                            }
                        }
                    }
                }
                Message::Ping(payload) => {
                    write.send(Message::Pong(payload)).await?;
                }
                Message::Close(_) => {
                    info!("WebSocket connection closed");
                    break;
                }
                _ => {}
            }
        }
        
        Ok(())
    }
    
    fn parse_ticker_message(&self, data: &Value) -> Option<MarketData> {
        let ticker_data = data["data"].as_array()?.get(0)?;
        
        let symbol = ticker_data["instId"].as_str()?.to_string();
        let price = ticker_data["last"].as_str()?.parse::<f64>().ok()?;
        let volume = ticker_data["vol24h"].as_str()?.parse::<f64>().unwrap_or(0.0);
        let bid = ticker_data["bidPx"].as_str()?.parse::<f64>().unwrap_or(price);
        let ask = ticker_data["askPx"].as_str()?.parse::<f64>().unwrap_or(price);
        
        let timestamp = ticker_data["ts"].as_str()?.parse::<u64>().ok()?;
        
        Some(MarketData {
            symbol,
            price,
            volume,
            bid,
            ask,
            timestamp,
            order_book: None,
        })
    }
}
