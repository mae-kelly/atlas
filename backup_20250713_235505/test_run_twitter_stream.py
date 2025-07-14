import asyncio
from data_streams.social_sentiment.twitter_stream import TwitterSentimentStream, TwitterSentimentAggregator
async def main():
    aggregator = TwitterSentimentAggregator()
    keywords = ['bitcoin', 'BTC', 'ethereum', 'ETH', 'crypto', 'cryptocurrency', 'DeFi', 'altcoin']
    try:
        twitter_stream = TwitterSentimentStream(
            keywords=keywords,
            on_sentiment=aggregator.process_sentiment
        )
        await twitter_stream.stream_tweets()
    except ValueError as e:
        print(f"❌ Setup Error: {e}")
        print("📋 Please set up your Twitter API credentials:")
        print("   1. Copy .env.template to .env")
        print("   2. Add your Twitter Bearer Token")
        print("   3. Run again")
    except KeyboardInterrupt:
        print("\n🛑 Stream stopped by user")
if __name__ == "__main__":
    asyncio.run(main())