import os
from typing import Dict, Any
from dotenv import load_dotenv
from loguru import logger
load_dotenv()
class EnvironmentConfig:
    """Environment configuration management"""
    def __init__(self):
        self.config = self._load_config()
        self._validate_critical_keys()
    def _load_config(self) -> Dict[str, Any]:
        """Load all configuration from environment variables"""
        return {
            'alpha_vantage_key': os.getenv('ALPHA_VANTAGE_API_KEY'),
            'twelve_data_key': os.getenv('TWELVE_DATA_API_KEY'),
            'polygon_key': os.getenv('POLYGON_API_KEY'),
            'finnhub_key': os.getenv('FINNHUB_API_KEY'),
            'iex_key': os.getenv('IEX_CLOUD_API_KEY'),
            'quandl_key': os.getenv('QUANDL_API_KEY'),
            'fred_key': os.getenv('FRED_API_KEY'),
            'binance_api_key': os.getenv('BINANCE_API_KEY'),
            'binance_secret': os.getenv('BINANCE_SECRET_KEY'),
            'coinbase_key': os.getenv('COINBASE_API_KEY'),
            'coinbase_secret': os.getenv('COINBASE_SECRET'),
            'coinbase_passphrase': os.getenv('COINBASE_PASSPHRASE'),
            'coingecko_key': os.getenv('COINGECKO_API_KEY'),
            'cryptocompare_key': os.getenv('CRYPTOCOMPARE_API_KEY'),
            'news_api_key': os.getenv('NEWS_API_KEY'),
            'twitter_bearer_token': os.getenv('TWITTER_BEARER_TOKEN'),
            'twitter_consumer_key': os.getenv('TWITTER_CONSUMER_KEY'),
            'twitter_consumer_secret': os.getenv('TWITTER_CONSUMER_SECRET'),
            'twitter_access_token': os.getenv('TWITTER_ACCESS_TOKEN'),
            'twitter_access_token_secret': os.getenv('TWITTER_ACCESS_TOKEN_SECRET'),
            'reddit_client_id': os.getenv('REDDIT_CLIENT_ID'),
            'reddit_client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
            'reddit_user_agent': os.getenv('REDDIT_USER_AGENT'),
            'database_url': os.getenv('DATABASE_URL'),
            'redis_url': os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'debug': os.getenv('DEBUG', 'false').lower() == 'true',
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        }
    def _validate_critical_keys(self):
        """Validate that critical API keys are present"""
        critical_keys = [
            'binance_api_key',  # At least one exchange
        ]
        missing_keys = []
        for key in critical_keys:
            if not self.config.get(key):
                missing_keys.append(key)
        if missing_keys:
            logger.warning(f"⚠️ Missing critical API keys: {', '.join(missing_keys)}")
            logger.warning("⚠️ Some features may not work properly")
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    def is_configured(self, service: str) -> bool:
        """Check if a service is properly configured"""
        service_requirements = {
            'binance': ['binance_api_key', 'binance_secret'],
            'twitter': ['twitter_bearer_token'],
            'news_api': ['news_api_key'],
            'reddit': ['reddit_client_id', 'reddit_client_secret'],
            'alpha_vantage': ['alpha_vantage_key'],
        }
        if service not in service_requirements:
            return False
        required_keys = service_requirements[service]
        return all(self.config.get(key) for key in required_keys)
    def get_configured_services(self) -> list:
        """Get list of properly configured services"""
        services = ['binance', 'twitter', 'news_api', 'reddit', 'alpha_vantage']
        return [service for service in services if self.is_configured(service)]
config = EnvironmentConfig()