import asyncio
import time
from core.fusion.data_fusion_engine import DataFusionEngine, PriceEvent, SentimentEvent, SignalLogger
from data_streams.price_feed.binance_ws_price_stream import BinancePriceStream
from data_streams.social_sentiment.twitter_stream import TwitterSentimentStream
class FusionIntegration:
    """
    Real-world integration connecting price streams + sentiment streams to fusion engine
    """
    def __init__(self):
        self.fusion_engine = DataFusionEngine(
            correlation_window=120,
            price_buffer_size=500,
            sentiment_buffer_size=200
        )
        self.signal_logger = SignalLogger()
        self.fusion_engine.add_signal_handler(self.signal_logger.log_signal)
        self.price_stream = None
        self.sentiment_stream = None
    async def handle_price_tick(self, tick_data: dict):
        """Convert Binance tick to PriceEvent and feed to fusion engine"""
        try:
            event = PriceEvent(
                symbol=tick_data.get("s"),
                price=float(tick_data.get("p", 0)),
                volume=float(tick_data.get("q", 0)),
                timestamp=tick_data.get("T", time.time() * 1000) / 1000  # Convert to seconds
            )
            await self.fusion_engine.ingest_price_event(event)
        except Exception as e:
            print(f"❌ Error processing price tick: {e}")
    async def handle_sentiment_data(self, sentiment_data: dict):
        """Convert Twitter sentiment to SentimentEvent and feed to fusion engine"""
        try:
            event = SentimentEvent(
                keywords=sentiment_data.get("keywords_matched", []),
                sentiment_score=sentiment_data["sentiment"]["vader"]["compound"],
                consensus=sentiment_data["sentiment"]["consensus"],
                confidence=abs(sentiment_data["sentiment"]["vader"]["compound"]),
                text_sample=sentiment_data.get("text", "")[:100],
                timestamp=sentiment_data.get("timestamp", time.time())
            )
            await self.fusion_engine.ingest_sentiment_event(event)
        except Exception as e:
            print(f"❌ Error processing sentiment data: {e}")
    async def start_real_time_fusion(self):
        """Start real-time price-sentiment fusion"""
        print("🧬 Starting Real-Time Price-Sentiment Fusion")
        print("=" * 50)
        try:
            self.price_stream = BinancePriceStream(
                pairs=["BTCUSDT", "ETHUSDT"],
                on_message=self.handle_price_tick
            )
            self.sentiment_stream = TwitterSentimentStream(
                keywords=['bitcoin', 'BTC', 'ethereum', 'ETH', 'crypto'],
                on_sentiment=self.handle_sentiment_data
            )
            await asyncio.gather(
                self.price_stream.connect(),
                self.sentiment_stream.stream_tweets()
            )
        except Exception as e:
            print(f"❌ Integration error: {e}")
            print("Make sure you have:")
            print("1. Twitter API credentials in .env")
            print("2. Internet connection")
            print("3. All dependencies installed")
async def main():
    integration = FusionIntegration()
    try:
        await integration.start_real_time_fusion()
    except KeyboardInterrupt:
        print("\n🛑 Real-time fusion stopped")
if __name__ == "__main__":
    asyncio.run(main())