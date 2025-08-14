import sys
import json
import time
import math
from typing import Dict, List

class RiskAnalyzer:
    def __init__(self):
        self.risk_factors = {
            'volatility': 0.25,
            'liquidity': 0.20,
            'market_cap': 0.15,
            'correlation': 0.15,
            'technical': 0.15,
            'fundamental': 0.10
        }
    
    def calculate_volatility_risk(self, symbol: str) -> float:
        """Calculate volatility-based risk score"""
        base_symbol = symbol.replace('-USDT', '').replace('-USD', '').replace('-USDC', '')
        
        # Major coins have lower volatility risk
        if base_symbol.upper() in ['BTC', 'ETH']:
            base_volatility = 0.3
        elif base_symbol.upper() in ['SOL', 'ADA', 'DOT']:
            base_volatility = 0.5
        else:
            base_volatility = 0.7
        
        # Add time-based variation
        current_time = int(time.time())
        time_factor = (current_time % 3600) / 3600
        volatility_risk = base_volatility + (time_factor - 0.5) * 0.2
        
        return max(0.0, min(1.0, volatility_risk))
    
    def calculate_liquidity_risk(self, symbol: str) -> float:
        """Calculate liquidity-based risk score"""
        base_symbol = symbol.replace('-USDT', '').replace('-USD', '').replace('-USDC', '')
        
        # Major trading pairs have lower liquidity risk
        if base_symbol.upper() in ['BTC', 'ETH', 'SOL']:
            return 0.1
        elif base_symbol.upper() in ['ADA', 'DOT', 'LINK', 'AVAX']:
            return 0.2
        elif base_symbol.upper() in ['MATIC', 'UNI', 'ATOM']:
            return 0.3
        else:
            return 0.6
    
    def calculate_market_cap_risk(self, symbol: str) -> float:
        """Calculate market cap based risk score"""
        base_symbol = symbol.replace('-USDT', '').replace('-USD', '').replace('-USDC', '')
        
        # Simulate market cap rankings
        market_cap_ranks = {
            'BTC': 1, 'ETH': 2, 'SOL': 5, 'ADA': 8, 'DOT': 12,
            'LINK': 15, 'AVAX': 18, 'MATIC': 20, 'UNI': 25, 'ATOM': 30
        }
        
        rank = market_cap_ranks.get(base_symbol.upper(), 100)
        
        if rank <= 10:
            return 0.1
        elif rank <= 30:
            return 0.3
        elif rank <= 100:
            return 0.5
        else:
            return 0.8
    
    def calculate_correlation_risk(self, symbol: str) -> float:
        """Calculate correlation risk with market"""
        base_symbol = symbol.replace('-USDT', '').replace('-USD', '').replace('-USDC', '')
        
        # Most altcoins are highly correlated with BTC
        if base_symbol.upper() == 'BTC':
            return 0.0  # BTC is the market
        elif base_symbol.upper() == 'ETH':
            return 0.2  # ETH has some independence
        else:
            return 0.7  # Most altcoins highly correlated
    
    def calculate_technical_risk(self, symbol: str) -> float:
        """Calculate technical analysis based risk"""
        # Simulate technical risk based on current "market conditions"
        current_time = int(time.time())
        market_phase = (current_time // 1800) % 4  # 30-minute cycles
        
        symbol_hash = hash(symbol) % 100
        
        if market_phase == 0:  # Trending up
            return 0.2 + symbol_hash / 1000
        elif market_phase == 1:  # Sideways
            return 0.4 + symbol_hash / 500
        elif market_phase == 2:  # Trending down
            return 0.6 + symbol_hash / 333
        else:  # High volatility
            return 0.8 + symbol_hash / 500
    
    def calculate_fundamental_risk(self, symbol: str) -> float:
        """Calculate fundamental analysis based risk"""
        base_symbol = symbol.replace('-USDT', '').replace('-USD', '').replace('-USDC', '')
        
        # Strong fundamental projects have lower risk
        strong_fundamentals = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'LINK']
        moderate_fundamentals = ['AVAX', 'MATIC', 'UNI', 'ATOM', 'FTM', 'NEAR']
        
        if base_symbol.upper() in strong_fundamentals:
            return 0.2
        elif base_symbol.upper() in moderate_fundamentals:
            return 0.4
        else:
            return 0.6
    
    def calculate_composite_risk_score(self, symbol: str) -> Dict:
        """Calculate comprehensive risk score"""
        
        risk_components = {
            'volatility': self.calculate_volatility_risk(symbol),
            'liquidity': self.calculate_liquidity_risk(symbol),
            'market_cap': self.calculate_market_cap_risk(symbol),
            'correlation': self.calculate_correlation_risk(symbol),
            'technical': self.calculate_technical_risk(symbol),
            'fundamental': self.calculate_fundamental_risk(symbol)
        }
        
        # Calculate weighted composite score
        composite_risk = sum(
            risk_components[factor] * self.risk_factors[factor]
            for factor in risk_components
        )
        
        # Risk level classification
        if composite_risk <= 0.3:
            risk_level = "LOW"
        elif composite_risk <= 0.5:
            risk_level = "MODERATE"
        elif composite_risk <= 0.7:
            risk_level = "HIGH"
        else:
            risk_level = "VERY_HIGH"
        
        return {
            'symbol': symbol,
            'composite_risk_score': round(composite_risk, 4),
            'risk_level': risk_level,
            'risk_components': risk_components,
            'recommendations': self.generate_risk_recommendations(composite_risk, risk_components),
            'timestamp': int(time.time())
        }
    
    def generate_risk_recommendations(self, composite_risk: float, components: Dict) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        if composite_risk > 0.7:
            recommendations.append("Consider reducing position size")
            recommendations.append("Use tighter stop losses")
            recommendations.append("Monitor position closely")
        
        if components['volatility'] > 0.6:
            recommendations.append("High volatility detected - use smaller leverage")
        
        if components['liquidity'] > 0.5:
            recommendations.append("Low liquidity - be cautious with large orders")
        
        if components['correlation'] > 0.6:
            recommendations.append("High market correlation - diversify portfolio")
        
        if components['technical'] > 0.6:
            recommendations.append("Negative technical signals - consider waiting")
        
        if not recommendations:
            recommendations.append("Risk levels acceptable for trading")
        
        return recommendations

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No symbol provided"}))
        return
    
    symbol = sys.argv[1]
    analyzer = RiskAnalyzer()
    
    try:
        risk_analysis = analyzer.calculate_composite_risk_score(symbol)
        print(json.dumps(risk_analysis, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()
