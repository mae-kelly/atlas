import asyncio
import json
from typing import Dict, List
from dataclasses import asdict
from loguru import logger
from collections import deque
import time
import numpy as np
class AlphaPredictionHandler:
    """
    Handles and analyzes alpha predictions
    """
    def __init__(self, max_predictions: int = 1000):
        self.predictions = deque(maxlen=max_predictions)
        self.performance_tracker = {}
    async def handle_prediction(self, prediction):
        """Handle incoming alpha prediction"""
        self.predictions.append(prediction)
        self._log_prediction(prediction)
        self._track_prediction_performance(prediction)
        await self._analyze_prediction_quality(prediction)
    def _log_prediction(self, prediction):
        """Log prediction details"""
        direction = "📈" if prediction.predicted_return > 0 else "📉"
        logger.info(f"{direction} ALPHA SIGNAL: {prediction.symbol}")
        logger.info(f"   Predicted Return: {prediction.predicted_return:.4f}")
        logger.info(f"   Confidence: {prediction.confidence:.2f}")
        logger.info(f"   Horizon: {prediction.prediction_horizon} minutes")
        logger.info(f"   Top Features: {self._get_top_features(prediction.contributing_features)}")
    def _get_top_features(self, features: Dict[str, float], top_n: int = 3) -> str:
        """Get top contributing features"""
        sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = sorted_features[:top_n]
        return ", ".join([f"{name}: {value:.3f}" for name, value in top_features])
    def _track_prediction_performance(self, prediction):
        """Track prediction for future performance analysis"""
        symbol = prediction.symbol
        if symbol not in self.performance_tracker:
            self.performance_tracker[symbol] = {
                'predictions': [],
                'hit_rate': 0.0,
                'avg_confidence': 0.0,
                'total_predictions': 0
            }
        self.performance_tracker[symbol]['predictions'].append({
            'timestamp': prediction.timestamp,
            'predicted_return': prediction.predicted_return,
            'confidence': prediction.confidence,
            'horizon_minutes': prediction.prediction_horizon,
            'verified': False,
            'actual_return': None
        })
        self.performance_tracker[symbol]['total_predictions'] += 1
    async def _analyze_prediction_quality(self, prediction):
        """Analyze the quality of the prediction"""
        model_consensus = prediction.model_consensus
        if len(model_consensus) > 1:
            predictions = list(model_consensus.values())
            consensus_std = np.std(predictions)
            consensus_mean = np.mean(predictions)
            if consensus_std / (abs(consensus_mean) + 1e-6) > 0.5:
                logger.warning(f"⚠️  High model disagreement for {prediction.symbol}: std={consensus_std:.4f}")
        if abs(prediction.predicted_return) > 0.01 and prediction.confidence < 0.3:
            logger.warning(f"⚠️  Large prediction with low confidence: {prediction.symbol}")
        if abs(prediction.predicted_return) > 0.05:
            logger.info(f"🚨 EXTREME ALPHA DETECTED: {prediction.symbol} - {prediction.predicted_return:.2%}")
    def get_prediction_summary(self) -> Dict:
        """Get summary of recent predictions"""
        if not self.predictions:
            return {"message": "No predictions yet"}
        recent_predictions = list(self.predictions)
        predicted_returns = [p.predicted_return for p in recent_predictions]
        confidences = [p.confidence for p in recent_predictions]
        symbol_counts = {}
        for pred in recent_predictions:
            symbol_counts[pred.symbol] = symbol_counts.get(pred.symbol, 0) + 1
        bullish_predictions = sum(1 for p in recent_predictions if p.predicted_return > 0)
        bearish_predictions = sum(1 for p in recent_predictions if p.predicted_return < 0)
        return {
            "total_predictions": len(recent_predictions),
            "bullish_predictions": bullish_predictions,
            "bearish_predictions": bearish_predictions,
            "avg_predicted_return": np.mean(predicted_returns),
            "avg_confidence": np.mean(confidences),
            "symbol_distribution": symbol_counts,
            "return_distribution": {
                "min": min(predicted_returns),
                "max": max(predicted_returns),
                "std": np.std(predicted_returns)
            },
            "recent_predictions": [
                {
                    "symbol": p.symbol,
                    "predicted_return": p.predicted_return,
                    "confidence": p.confidence,
                    "timestamp": p.timestamp
                }
                for p in recent_predictions[-5:]
            ]
        }
    def verify_predictions_with_actual_data(self, symbol: str, actual_returns: List[Dict]):
        """
        Verify past predictions with actual market data
        Format: [{"timestamp": ts, "return": actual_return}, ...]
        """
        if symbol not in self.performance_tracker:
            return
        tracker = self.performance_tracker[symbol]
        verified_count = 0
        correct_predictions = 0
        for prediction in tracker['predictions']:
            if prediction['verified']:
                continue
            prediction_time = prediction['timestamp']
            horizon_seconds = prediction['horizon_minutes'] * 60
            target_time = prediction_time + horizon_seconds
            closest_actual = min(
                actual_returns,
                key=lambda x: abs(x['timestamp'] - target_time),
                default=None
            )
            if closest_actual and abs(closest_actual['timestamp'] - target_time) < 300:  # Within 5 minutes
                prediction['actual_return'] = closest_actual['return']
                prediction['verified'] = True
                verified_count += 1
                predicted_direction = 1 if prediction['predicted_return'] > 0 else -1
                actual_direction = 1 if closest_actual['return'] > 0 else -1
                if predicted_direction == actual_direction:
                    correct_predictions += 1
        if verified_count > 0:
            tracker['hit_rate'] = correct_predictions / verified_count
            logger.info(f"📊 {symbol} Prediction Performance: {correct_predictions}/{verified_count} = {tracker['hit_rate']:.2%}")