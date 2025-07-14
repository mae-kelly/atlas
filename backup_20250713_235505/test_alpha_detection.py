import asyncio
import time
import random
import numpy as np
from ml.alpha_detection.alpha_detection_engine import AlphaDetectionEngine
from ml.alpha_detection.predictors.alpha_prediction_handler import AlphaPredictionHandler
class MockFusionSignalGenerator:
    """
    Generate mock fusion signals for testing alpha detection
    """
    def __init__(self, alpha_engine: AlphaDetectionEngine):
        self.alpha_engine = alpha_engine
        self.running = False
        self.price_history = [45000.0]
        self.sentiment_history = []
    async def start_generating(self):
        """Start generating mock fusion signals"""
        self.running = True
        while self.running:
            signal_data = self._generate_mock_signal()
            await self.alpha_engine.process_fusion_signal(signal_data)
            await asyncio.sleep(10)
    def _generate_mock_signal(self) -> dict:
        """Generate a mock fusion signal with realistic data"""
        price_change = random.gauss(0, 0.01)
        new_price = self.price_history[-1] * (1 + price_change)
        self.price_history.append(new_price)
        if len(self.price_history) > 100:
            self.price_history.pop(0)
        if len(self.price_history) >= 10:
            price_momentum = (self.price_history[-1] - self.price_history[-10]) / self.price_history[-10]
        else:
            price_momentum = price_change
        base_sentiment = random.gauss(0, 0.3)
        if random.random() < 0.4:
            sentiment_correlation = price_momentum * random.uniform(0.3, 0.8)
            base_sentiment += sentiment_correlation
        sentiment_score = max(-1, min(1, base_sentiment))
        self.sentiment_history.append(sentiment_score)
        if len(self.sentiment_history) > 50:
            self.sentiment_history.pop(0)
        if len(self.sentiment_history) >= 5:
            sentiment_momentum = np.mean(self.sentiment_history[-5:]) - np.mean(self.sentiment_history[-10:-5]) if len(self.sentiment_history) >= 10 else 0
        else:
            sentiment_momentum = 0
        if len(self.sentiment_history) >= 10 and len(self.price_history) >= 10:
            price_returns = np.diff(self.price_history[-10:]) / self.price_history[-10:-1]
            sentiment_recent = self.sentiment_history[-9:]
            correlation = np.corrcoef(price_returns, sentiment_recent)[0, 1] if not np.isnan(np.corrcoef(price_returns, sentiment_recent)[0, 1]) else 0
        else:
            correlation = random.uniform(-0.5, 0.5)
        future_return = random.gauss(price_momentum * 0.5 + sentiment_score * 0.2, 0.005)
        return {
            'symbol': 'BTCUSDT',
            'price_data': {
                'prices': self.price_history.copy(),
                'momentum': price_momentum,
                'volatility': np.std(np.diff(self.price_history[-20:]) / self.price_history[-20:-1]) if len(self.price_history) >= 20 else 0.01,
                'volume_momentum': random.uniform(0.5, 2.0)
            },
            'sentiment_data': {
                'sentiment_score': sentiment_score,
                'momentum': sentiment_momentum,
                'volatility': np.std(self.sentiment_history[-10:]) if len(self.sentiment_history) >= 10 else 0.2,
                'consensus': 'bullish' if sentiment_score > 0.1 else 'bearish' if sentiment_score < -0.1 else 'neutral',
                'volume': random.uniform(10, 100)
            },
            'correlation_data': {
                'correlation': correlation,
                'stability': random.uniform(0.3, 0.9)
            },
            'signal_strength': random.uniform(0.2, 0.9),
            'actual_return': future_return,  # This would come from actual market data
            'timestamp': time.time()
        }
    def stop(self):
        """Stop generating signals"""
        self.running = False
async def main():
    print("🤖 Testing Alpha Detection Engine")
    print("=" * 50)
    alpha_engine = AlphaDetectionEngine(
        prediction_horizon=15,
        feature_window=60,
        retrain_interval=120
    )
    prediction_handler = AlphaPredictionHandler()
    alpha_engine.add_alpha_handler(prediction_handler.handle_prediction)
    signal_generator = MockFusionSignalGenerator(alpha_engine)
    try:
        print("🚀 Starting alpha detection test...")
        print("📊 Generating mock fusion signals...")
        print("🧠 Engine will retrain every 2 minutes")
        print("Press Ctrl+C to stop\n")
        generator_task = asyncio.create_task(signal_generator.start_generating())
        for i in range(30):
            await asyncio.sleep(10)
            if i % 6 == 5:
                print(f"\n📈 SUMMARY at {(i+1)*10} seconds:")
                alpha_summary = alpha_engine.get_performance_summary()
                if "models_trained" in alpha_summary:
                    print(f"   Models Trained: {alpha_summary['models_trained']}")
                    print(f"   Training Samples: {alpha_summary['training_samples']}")
                    print(f"   Predictions Made: {alpha_summary['predictions_made']}")
                    print(f"   Best Model: {alpha_summary['best_model']}")
                pred_summary = prediction_handler.get_prediction_summary()
                if "total_predictions" in pred_summary:
                    print(f"   Total Predictions: {pred_summary['total_predictions']}")
                    print(f"   Avg Confidence: {pred_summary['avg_confidence']:.2f}")
                    print(f"   Bullish/Bearish: {pred_summary['bullish_predictions']}/{pred_summary['bearish_predictions']}")
        signal_generator.stop()
        print("\n" + "=" * 50)
        print("🎯 FINAL ALPHA DETECTION SUMMARY")
        print("=" * 50)
        final_alpha_summary = alpha_engine.get_performance_summary()
        final_pred_summary = prediction_handler.get_prediction_summary()
        print("🤖 Alpha Engine Performance:")
        for key, value in final_alpha_summary.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for k, v in value.items():
                    print(f"     {k}: {v}")
            else:
                print(f"   {key}: {value}")
        print("\n📊 Prediction Summary:")
        for key, value in final_pred_summary.items():
            if key != "recent_predictions":
                print(f"   {key}: {value}")
        print("\n🔮 Recent Predictions:")
        for pred in final_pred_summary.get("recent_predictions", [])[-3:]:
            direction = "📈" if pred["predicted_return"] > 0 else "📉"
            print(f"   {direction} {pred['symbol']}: {pred['predicted_return']:.4f} (conf: {pred['confidence']:.2f})")
    except KeyboardInterrupt:
        print("\n🛑 Test stopped by user")
        signal_generator.stop()
if __name__ == "__main__":
    asyncio.run(main())