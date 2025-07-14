"""
Basic API Test - Test core functionality without complex dependencies
"""
import asyncio
import aiohttp
import os
from dotenv import load_dotenv
load_dotenv()
async def test_free_apis():
    """Test APIs that don't require authentication"""
    print("🧪 Testing Free APIs...")
    print("=" * 40)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd') as response:
                if response.status == 200:
                    data = await response.json()
                    btc_price = data['bitcoin']['usd']
                    print(f"✅ CoinGecko: Bitcoin price ${btc_price:,.2f}")
                else:
                    print(f"❌ CoinGecko: HTTP {response.status}")
    except Exception as e:
        print(f"❌ CoinGecko: {e}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT') as response:
                if response.status == 200:
                    data = await response.json()
                    btc_price = float(data['price'])
                    print(f"✅ Binance: Bitcoin price ${btc_price:,.2f}")
                else:
                    print(f"❌ Binance: HTTP {response.status}")
    except Exception as e:
        print(f"❌ Binance: {e}")
    try:
        import feedparser
        feed = feedparser.parse('https://feeds.reuters.com/reuters/businessNews')
        if feed.entries:
            print(f"✅ Reuters RSS: {len(feed.entries)} articles fetched")
        else:
            print("❌ Reuters RSS: No articles found")
    except Exception as e:
        print(f"❌ Reuters RSS: {e}")
    print("=" * 40)
    print("🎯 Basic API test completed!")
async def test_authenticated_apis():
    """Test APIs that require authentication"""
    print("\n🔐 Testing Authenticated APIs...")
    print("=" * 40)
    news_api_key = os.getenv('NEWS_API_KEY')
    if news_api_key:
        try:
            async with aiohttp.ClientSession() as session:
                url = 'https://newsapi.org/v2/everything'
                params = {
                    'q': 'bitcoin',
                    'pageSize': 5,
                    'apiKey': news_api_key
                }
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        article_count = len(data.get('articles', []))
                        print(f"✅ News API: {article_count} articles fetched")
                    else:
                        print(f"❌ News API: HTTP {response.status}")
        except Exception as e:
            print(f"❌ News API: {e}")
    else:
        print("⚠️ News API: Key not configured")
    twitter_token = os.getenv('TWITTER_BEARER_TOKEN')
    if twitter_token:
        try:
            async with aiohttp.ClientSession() as session:
                headers = {'Authorization': f'Bearer {twitter_token}'}
                url = 'https://api.twitter.com/2/tweets/search/recent'
                params = {'query': 'bitcoin', 'max_results': 10}
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        tweet_count = len(data.get('data', []))
                        print(f"✅ Twitter API: {tweet_count} tweets fetched")
                    else:
                        print(f"❌ Twitter API: HTTP {response.status}")
        except Exception as e:
            print(f"❌ Twitter API: {e}")
    else:
        print("⚠️ Twitter API: Bearer token not configured")
    print("=" * 40)
def check_environment():
    """Check environment setup"""
    print("🌍 Environment Check...")
    print("=" * 40)
    if os.path.exists('.env'):
        print("✅ .env file found")
    else:
        print("❌ .env file not found")
        print("   Please copy config/api_credentials.env to .env and configure your API keys")
    required_packages = ['aiohttp', 'python-dotenv', 'feedparser']
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} not installed")
            print(f"   Run: pip install {package}")
    print("=" * 40)
async def main():
    """Main test function"""
    print("🚀 AI Trading Empire - Basic API Test")
    print("=" * 50)
    check_environment()
    await test_free_apis()
    await test_authenticated_apis()
    print("\n✨ Test completed!")
    print("\nNext steps:")
    print("1. Configure API keys in .env file")
    print("2. Run: python scripts/check_api_status.py")
    print("3. Start the trading system")
if __name__ == "__main__":
    asyncio.run(main())