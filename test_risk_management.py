import asyncio
import time
import random
import numpy as np
from risk_management.risk_management_engine import RiskManagementEngine, PositionSizeRecommendation
from ml.alpha_detection.alpha_detection_engine import AlphaPrediction

class MockAlphaPrediction:
    """Mock alpha prediction for testing"""
    def __init__(self, symbol: str, predicted_return: float, confidence: float):
        self.symbol = symbol
        self.predicted_return = predicted_return
        self.confidence = confidence
        self.prediction_horizon = 15  # minutes
        self.timestamp = time.time()
        self.metadata = {
            'volatility_estimate': abs(predicted_return) * random.uniform(2, 5),
            'model_count': 5
        }

class RiskManagementTester:
    """Test the risk management engine"""
    
    def __init__(self):
        self.risk_engine = RiskManagementEngine(
            base_capital=100000.0,
            max_portfolio_risk=0.02,
            max_single_position=0.10,
            target_sharpe=1.5
        )
        
        self.position_recommendations = []
        
    async def test_position_sizing(self):
        """Test position sizing for various alpha predictions"""
        print("🧪 Testing Position Sizing Logic")
        print("=" * 40)
        
        # Test scenarios
        test_predictions = [
            MockAlphaPrediction("BTCUSDT", 0.02, 0.8),    # High confidence bullish
            MockAlphaPrediction("ETHUSDT", -0.015, 0.7),  # High confidence bearish
            MockAlphaPrediction("BTCUSDT", 0.005, 0.4),   # Low confidence bullish
            MockAlphaPrediction("ETHUSDT", 0.03, 0.9),    # Very high confidence, large move
            MockAlphaPrediction("ADAUSDT", -0.001, 0.3),  # Very low confidence
        ]
        
        for i, prediction in enumerate(test_predictions):
            print(f"\n📊 Test Case {i+1}: {prediction.symbol}")
            print(f"   Expected Return: {prediction.predicted_return:.2%}")
            print(f"   Confidence: {prediction.confidence:.1%}")
            
            # Get position size recommendation
            recommendation = await self.risk_engine.evaluate_position_size(prediction)
            self.position_recommendations.append(recommendation)
            
            # Simulate taking the position
            if recommendation.recommended_size > 0.001:  # Only take significant positions
                entry_price = random.uniform(40000, 50000)  # Mock price
                self.risk_engine.update_position(
                    prediction.symbol, 
                    recommendation.recommended_size, 
                    entry_price
                )
                
                print(f"   ✅ Position taken: {recommendation.recommended_size:.2%}")
            else:
                print(f"   ❌ Position too small: {recommendation.recommended_size:.2%}")
            
            await asyncio.sleep(1)  # Brief pause between tests
    
    async def test_portfolio_risk_monitoring(self):
        """Test portfolio risk monitoring"""
        print("\n🛡️ Testing Portfolio Risk Monitoring")
        print("=" * 40)
        
        # Simulate price movements and update P&L
        symbols = list(self.risk_engine.positions.keys())
        
        for _ in range(10):  # 10 price updates
            for symbol in symbols:
                # Simulate price movement
                current_position = self.risk_engine.positions[symbol]
                entry_price = current_position['entry_price']
                
                # Random price movement
                price_change = random.gauss(0, 0.02)  # 2% volatility
                new_price = entry_price * (1 + price_change)
                
                # Update position P&L
                self.risk_engine.update_position_pnl(symbol, new_price)
            
            # Check portfolio heat
            portfolio_heat = await self.risk_engine._calculate_portfolio_heat()
            print(f"Portfolio Heat: {portfolio_heat:.1%}")
            
            await asyncio.sleep(0.5)
    
    def analyze_position_recommendations(self):
        """Analyze the position recommendations"""
        print("\n📈 Position Sizing Analysis")
        print("=" * 40)
        
        if not self.position_recommendations:
            print("No position recommendations to analyze")
            return
        
        # Analyze recommendation patterns
        sizes = [rec.recommended_size for rec in self.position_recommendations]
        kelly_fractions = [rec.kelly_fraction for rec in self.position_recommendations]
        confidence_adjustments = [rec.confidence_adjustment for rec in self.position_recommendations]
        expected_returns = [rec.expected_return for rec in self.position_recommendations]
        
        print(f"Average Position Size: {np.mean(sizes):.2%}")
        print(f"Max Position Size: {np.max(sizes):.2%}")
        print(f"Min Position Size: {np.min(sizes):.2%}")
        print(f"Average Kelly Fraction: {np.mean(kelly_fractions):.2%}")
        print(f"Average Confidence Adjustment: {np.mean(confidence_adjustments):.2f}")
        
        # Correlation analysis
        if len(expected_returns) > 1:
            size_return_corr = np.corrcoef(sizes, expected_returns)[0, 1]
            print(f"Size-Return Correlation: {size_return_corr:.2f}")
        
        # Print individual recommendations
        print("\n📋 Individual Recommendations:")
        for rec in self.position_recommendations:
            direction = "📈" if rec.expected_return > 0 else "📉"
            print(f"{direction} {rec.symbol}: Size={rec.recommended_size:.2%}, "
                  f"Kelly={rec.kelly_fraction:.2%}, Return={rec.expected_return:.2%}")
    
    def test_kelly_criterion(self):
        """Test Kelly criterion calculations"""
        print("\n📐 Testing Kelly Criterion")
        print("=" * 40)
        
        from risk_management.position_sizing.kelly_criterion import KellyCriterion
        
        kelly_calc = KellyCriterion()
        
        # Test scenarios
        test_cases = [
            (0.6, 0.02, 0.01),  # 60% win, 2% gain, 1% loss
            (0.55, 0.015, 0.015), # 55% win, equal gain/loss
            (0.7, 0.01, 0.02),  # 70% win, smaller gains than losses
            (0.4, 0.05, 0.02),  # 40% win (should be negative Kelly)
        ]
        
        for i, (win_prob, win_amount, loss_amount) in enumerate(test_cases):
            kelly_fraction = kelly_calc.calculate_kelly_fraction(win_prob, win_amount, loss_amount)
            continuous_kelly = kelly_calc.calculate_continuous_kelly(
                win_prob * win_amount - (1 - win_prob) * loss_amount,
                (win_amount + loss_amount) ** 2 * win_prob * (1 - win_prob)
            )
            
            print(f"Case {i+1}: Win={win_prob:.0%}, Gain={win_amount:.1%}, Loss={loss_amount:.1%}")
            print(f"   Kelly Fraction: {kelly_fraction:.2%}")
            print(f"   Continuous Kelly: {continuous_kelly:.2%}")
            
            # Test fractional Kelly
            fractional = kelly_calc.calculate_fractional_kelly(kelly_fraction, 0.25)
            print(f"   Quarter Kelly: {fractional:.2%}")

async def main():
    print("🛡️ Risk Management Engine Test Suite")
    print("=" * 50)
    
    tester = RiskManagementTester()
    
    try:
        # Test position sizing
        await tester.test_position_sizing()
        
        # Test portfolio monitoring
        await tester.test_portfolio_risk_monitoring()
        
        # Analyze results
        tester.analyze_position_recommendations()
        
        # Test Kelly criterion
        tester.test_kelly_criterion()
        
        # Final portfolio summary
        print("\n📊 Final Portfolio Summary")
        print("=" * 40)
        portfolio_summary = tester.risk_engine.get_portfolio_summary()
        
        for key, value in portfolio_summary.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
        
        print("\n✅ Risk Management Test Complete!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
