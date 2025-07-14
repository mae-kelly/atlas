"""
API Status Checker - Verify all APIs are working correctly
"""
import asyncio
import aiohttp
import time
from typing import Dict
import os
from dotenv import load_dotenv
load_dotenv()
class APIStatusChecker:
    def __init__(self):
        self.results = {}
    async def check_all_apis(self) -> Dict:
        """Check status of all configured APIs"""
        apis_to_check = [
            ('Binance', self._check_binance),
            ('CoinGecko', self._check_coingecko), 
            ('News API', self._check_news_api),
            ('Twitter API', self._check_twitter),
            ('Alpha Vantage', self._check_alpha_vantage),
        ]
        print("🔍 Checking API Status...")
        print("=" * 50)
        for api_name, check_func in apis_to_check:
            try:
                status = await check_func()
                self.results[api_name] = status
                status_icon = "✅" if status['working'] else "❌"
                print(f"{status_icon} {api_name}: {status['message']}")
            except Exception as e:
                self.results[api_name] = {'working': False, 'message': f'Error: {e}'}
                print(f"❌ {api_name}: Error - {e}")
        print("=" * 50)
        working_apis = sum(1 for result in self.results.values() if result['working'])
        total_apis = len(self.results)
        print(f"📊 Summary: {working_apis}/{total_apis} APIs working")
        if working_apis >= total_apis * 0.5:
            print("🎉 System ready for testing!")
        else:
            print("⚠️  Some APIs need configuration")
        return self.results
    async def _check_binance(self) -> Dict:
        """Check Binance API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.binance.com/api/v3/ping') as response:
                    if response.status == 200:
                        return {'working': True, 'message': 'API accessible'}
                    else:
                        return {'working': False, 'message': f'HTTP {response.status}'}
        except Exception as e:
            return {'working': False, 'message': f'Connection failed: {e}'}
    async def _check_coingecko(self) -> Dict:
        """Check CoinGecko API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.coingecko.com/api/v3/ping') as response:
                    if response.status == 200:
                        return {'working': True, 'message': 'API accessible'}
                    else:
                        return {'working': False, 'message': f'HTTP {response.status}'}
        except Exception as e:
            return {'working': False, 'message': f'Connection failed: {e}'}
    async def _check_news_api(self) -> Dict:
        """Check News API"""
        api_key = os.getenv('NEWS_API_KEY')
        if not api_key:
            return {'working': False, 'message': 'API key not configured'}
        try:
            async with aiohttp.ClientSession() as session:
                url = 'https://newsapi.org/v2/top-headlines'
                params = {'country': 'us', 'pageSize': 1, 'apiKey': api_key}
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return {'working': True, 'message': 'API working with key'}
                    else:
                        return {'working': False, 'message': f'HTTP {response.status}'}
        except Exception as e:
            return {'working': False, 'message': f'Connection failed: {e}'}
    async def _check_twitter(self) -> Dict:
        """Check Twitter API"""
        bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        if not bearer_token:
            return {'working': False, 'message': 'Bearer token not configured'}
        try:
            async with aiohttp.ClientSession() as session:
                headers = {'Authorization': f'Bearer {bearer_token}'}
                url = 'https://api.twitter.com/2/tweets/search/recent'
                params = {'query': 'bitcoin', 'max_results': 10}
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        return {'working': True, 'message': 'API working with token'}
                    else:
                        return {'working': False, 'message': f'HTTP {response.status}'}
        except Exception as e:
            return {'working': False, 'message': f'Connection failed: {e}'}
    async def _check_alpha_vantage(self) -> Dict:
        """Check Alpha Vantage API"""
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            return {'working': False, 'message': 'API key not configured'}
        try:
            async with aiohttp.ClientSession() as session:
                url = 'https://www.alphavantage.co/query'
                params = {
                    'function': 'GLOBAL_QUOTE',
                    'symbol': 'AAPL',
                    'apikey': api_key
                }
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'Global Quote' in data:
                            return {'working': True, 'message': 'API working with key'}
                        else:
                            return {'working': False, 'message': 'Invalid response or rate limited'}
                    else:
                        return {'working': False, 'message': f'HTTP {response.status}'}
        except Exception as e:
            return {'working': False, 'message': f'Connection failed: {e}'}
async def main():
    checker = APIStatusChecker()
    results = await checker.check_all_apis()
    import json
    with open('api_status_report.json', 'w') as f:
        json.dump({
            'timestamp': time.time(),
            'results': results
        }, f, indent=2)
    print(f"\n📄 Detailed report saved to: api_status_report.json")
if __name__ == "__main__":
    asyncio.run(main())