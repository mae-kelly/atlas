import asyncio
import time
import random
from core.fusion.data_fusion_engine import DataFusionEngine, PriceEvent, SentimentEvent, SignalLogger
class MockDataFeeder:
    """
    Simulates price and sentiment data for testing the fusion engine
    """
    def __init__(self, fusion_engine: DataFusionEngine):
        self.fusion_engine = fusion_engine
        self.running = False
    async def start_feeding(self):
        """Start feeding mock data"""
        self.running = True
        await asyncio.gather(
            self._feed_price_data(),
            self._feed_sentiment_data()
        )
    async def _feed_price_data(self):
        """Feed mock price data"""
        price = 45000.0
        while self.running:
            change_pct = random.gauss(0, 0.002)
            price *= (1 + change_pct)
            volume = random.uniform(0.5, 2.0)
            event = PriceEvent(
                symbol="BTCUSDT",
                price=price,
                volume=volume,
                timestamp=time.time()
            )
            await self.fusion_engine.ingest_price_event(event)
            await asyncio.sleep(2)
    async def _feed_sentiment_data(self):
        """Feed mock sentiment data"""
        while self.running:
            base_sentiment = random.gauss(0, 0.3)
            if random.random() < 0.3:
                base_sentiment += random.gauss(0, 0.4)
            sentiment_score = max(-1, min(1, base_sentiment))
            consensus = (
                "bullish" if sentiment_score > 0.1 
                else "bearish" if sentiment_score < -0.1 
                else "neutral"
            )
            event = SentimentEvent(
                keywords=["bitcoin", "BTC"],
                sentiment_score=sentiment_score,
                consensus=consensus,
                confidence=random.uniform(0.6, 0.9),
                text_sample=f"Mock tweet with {consensus} sentiment",
                timestamp=time.time()
            )
            await self.fusion_engine.ingest_sentiment_event(event)
            await asyncio.sleep(5)
    def stop(self):
        """Stop feeding data"""
        self.running = False
async def main():
    print("🧬 Testing Data Fusion Engine Integration")
    print("=" * 50)
    fusion_engine = DataFusionEngine(
        correlation_window=30,
        price_buffer_size=100,
        sentiment_buffer_size=50
    )
    signal_logger = SignalLogger()
    fusion_engine.add_signal_handler(signal_logger.log_signal)
    data_feeder = MockDataFeeder(fusion_engine)
    try:
        print("🚀 Starting mock data feeds...")
        print("📊 Watch for correlation signals...")
        print("Press Ctrl+C to stop\n")
        feed_task = asyncio.create_task(data_feeder.start_feeding())
        await asyncio.sleep(60)
        data_feeder.stop()
        print("\n" + "=" * 50)
        print("📈 FUSION ENGINE SUMMARY")
        print("=" * 50)
        correlation_summary = fusion_engine.get_correlation_summary()
        print(f"Average Correlation: {correlation_summary.get('avg_correlation', 0):.3f}")
        print(f"Total Signals Generated: {correlation_summary.get('total_signals', 0)}")
        print(f"Events Processed: {correlation_summary.get('events_processed', 0)}")
        signal_summary = signal_logger.get_signal_summary()
        print(f"\nSignal Types: {signal_summary.get('signal_types', {})}")
        print(f"Direction Distribution: {signal_summary.get('direction_distribution', {})}")
        print(f"Average Signal Strength: {signal_summary.get('avg_strength', 0):.3f}")
    except KeyboardInterrupt:
        print("\n🛑 Test stopped by user")
        data_feeder.stop()
if __name__ == "__main__":
    asyncio.run(main())