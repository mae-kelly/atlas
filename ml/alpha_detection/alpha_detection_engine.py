import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Optional, Tuple
import joblib
import time
from dataclasses import dataclass
from loguru import logger
import asyncio
from collections import deque
@dataclass
class AlphaPrediction:
    symbol: str
    predicted_return: float
    confidence: float
    prediction_horizon: int
    contributing_features: Dict[str, float]
    model_consensus: Dict[str, float]
    timestamp: float
    metadata: Dict
@dataclass
class TrainingMetrics:
    model_name: str
    mse: float
    mae: float
    r2: float
    cross_val_score: float
    feature_importance: Dict[str, float]
    training_samples: int
class AlphaDetectionEngine:
    """
    ML engine that learns patterns from fusion signals to predict alpha opportunities
    """
    def __init__(self, 
                 prediction_horizon: int = 15,
                 feature_window: int = 60,
                 retrain_interval: int = 3600):
        self.prediction_horizon = prediction_horizon
        self.feature_window = feature_window
        self.retrain_interval = retrain_interval
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.training_buffer = deque(maxlen=10000)
        self.last_retrain_time = 0
        self.prediction_history = deque(maxlen=1000)
        self.performance_metrics = {}
        self.alpha_handlers = []
        self._initialize_models()
        logger.info("🤖 Alpha Detection Engine initialized")
    def _initialize_models(self):
        """Initialize the model ensemble"""
        self.models = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
                verbose=-1
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'elastic_net': ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                random_state=42
            )
        }
        for model_name in self.models.keys():
            self.scalers[model_name] = RobustScaler()
    def add_alpha_handler(self, handler):
        """Add callback for alpha predictions"""
        self.alpha_handlers.append(handler)
    async def process_fusion_signal(self, fusion_signal_data: Dict):
        """
        Process incoming fusion signals and generate alpha predictions
        """
        try:
            features = self._extract_features(fusion_signal_data)
            self._add_training_example(features, fusion_signal_data)
            if self._should_retrain():
                await self._retrain_models()
            if self._models_are_trained():
                prediction = await self._generate_alpha_prediction(features, fusion_signal_data)
                if prediction:
                    await self._emit_alpha_prediction(prediction)
        except Exception as e:
            logger.error(f"❌ Error processing fusion signal: {e}")
    def _extract_features(self, signal_data: Dict) -> Dict[str, float]:
        """
        Extract ML features from fusion signal data
        """
        features = {}
        if 'price_data' in signal_data:
            price_data = signal_data['price_data']
            features.update({
                'price_momentum': price_data.get('momentum', 0),
                'price_volatility': price_data.get('volatility', 0),
                'price_rsi': self._calculate_rsi(price_data.get('prices', [])),
                'price_sma_ratio': self._calculate_sma_ratio(price_data.get('prices', [])),
                'volume_momentum': price_data.get('volume_momentum', 0)
            })
        if 'sentiment_data' in signal_data:
            sentiment_data = signal_data['sentiment_data']
            features.update({
                'sentiment_score': sentiment_data.get('sentiment_score', 0),
                'sentiment_momentum': sentiment_data.get('momentum', 0),
                'sentiment_volatility': sentiment_data.get('volatility', 0),
                'sentiment_consensus': self._encode_sentiment(sentiment_data.get('consensus', 'neutral')),
                'social_volume': sentiment_data.get('volume', 0)
            })
        if 'correlation_data' in signal_data:
            corr_data = signal_data['correlation_data']
            features.update({
                'price_sentiment_correlation': corr_data.get('correlation', 0),
                'correlation_strength': abs(corr_data.get('correlation', 0)),
                'correlation_stability': corr_data.get('stability', 0)
            })
        if 'signal_strength' in signal_data:
            features['signal_strength'] = signal_data['signal_strength']
        features.update(self._extract_time_features())
        if 'price_data' in signal_data and 'prices' in signal_data['price_data']:
            prices = signal_data['price_data']['prices']
            features.update(self._calculate_technical_indicators(prices))
        return features
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    def _calculate_sma_ratio(self, prices: List[float], short_period: int = 5, long_period: int = 20) -> float:
        """Calculate short MA / long MA ratio"""
        if len(prices) < long_period:
            return 1.0
        short_ma = np.mean(prices[-short_period:])
        long_ma = np.mean(prices[-long_period:])
        return short_ma / long_ma if long_ma != 0 else 1.0
    def _encode_sentiment(self, sentiment: str) -> float:
        """Encode sentiment string to numeric"""
        mapping = {'bearish': -1.0, 'neutral': 0.0, 'bullish': 1.0}
        return mapping.get(sentiment, 0.0)
    def _extract_time_features(self) -> Dict[str, float]:
        """Extract time-based features"""
        now = time.time()
        dt = pd.Timestamp.fromtimestamp(now)
        return {
            'hour_of_day': dt.hour / 24.0,
            'day_of_week': dt.dayofweek / 6.0,
            'is_weekend': float(dt.dayofweek >= 5),
            'is_market_hours': float(9 <= dt.hour <= 16)  # Rough market hours
        }
    def _calculate_technical_indicators(self, prices: List[float]) -> Dict[str, float]:
        """Calculate additional technical indicators"""
        if len(prices) < 20:
            return {}
        prices_array = np.array(prices)
        sma = np.mean(prices_array[-20:])
        std = np.std(prices_array[-20:])
        bb_upper = sma + (2 * std)
        bb_lower = sma - (2 * std)
        bb_position = (prices_array[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        momentum_5 = (prices_array[-1] - prices_array[-6]) / prices_array[-6] if len(prices) > 5 else 0
        momentum_10 = (prices_array[-1] - prices_array[-11]) / prices_array[-11] if len(prices) > 10 else 0
        return {
            'bb_position': bb_position,
            'momentum_5': momentum_5,
            'momentum_10': momentum_10,
            'volatility_20': std / sma if sma != 0 else 0
        }
    def _add_training_example(self, features: Dict[str, float], signal_data: Dict):
        """Add training example to buffer"""
        future_return = signal_data.get('actual_return', 0.0)
        training_example = {
            'features': features,
            'target': future_return,
            'timestamp': time.time(),
            'symbol': signal_data.get('symbol', 'BTCUSDT')
        }
        self.training_buffer.append(training_example)
    def _should_retrain(self) -> bool:
        """Check if models should be retrained"""
        return (
            time.time() - self.last_retrain_time > self.retrain_interval and
            len(self.training_buffer) >= 50
        )
    def _models_are_trained(self) -> bool:
        """Check if models have been trained"""
        return len(self.models) > 0 and all(
            hasattr(model, 'feature_importances_') or hasattr(model, 'coef_')
            for model in self.models.values()
        )
    async def _retrain_models(self):
        """Retrain all models with latest data"""
        logger.info("🔄 Retraining alpha detection models...")
        try:
            X, y = self._prepare_training_data()
            if len(X) < 30:
                logger.warning("❌ Insufficient training data")
                return
            training_metrics = {}
            for model_name, model in self.models.items():
                try:
                    X_scaled = self.scalers[model_name].fit_transform(X)
                    tscv = TimeSeriesSplit(n_splits=3)
                    cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='neg_mean_squared_error')
                    model.fit(X_scaled, y)
                    y_pred = model.predict(X_scaled)
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = dict(zip(self.feature_columns, model.feature_importances_))
                    elif hasattr(model, 'coef_'):
                        feature_importance = dict(zip(self.feature_columns, abs(model.coef_)))
                    else:
                        feature_importance = {}
                    metrics = TrainingMetrics(
                        model_name=model_name,
                        mse=mean_squared_error(y, y_pred),
                        mae=mean_absolute_error(y, y_pred),
                        r2=r2_score(y, y_pred),
                        cross_val_score=-cv_scores.mean(),
                        feature_importance=feature_importance,
                        training_samples=len(X)
                    )
                    training_metrics[model_name] = metrics
                    logger.info(f"✅ {model_name} trained - R²: {metrics.r2:.3f}, CV Score: {metrics.cross_val_score:.3f}")
                except Exception as e:
                    logger.error(f"❌ Error training {model_name}: {e}")
            self.performance_metrics = training_metrics
            self.last_retrain_time = time.time()
            await self._save_models()
        except Exception as e:
            logger.error(f"❌ Error in model retraining: {e}")
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from buffer"""
        df_data = []
        for example in self.training_buffer:
            row = example['features'].copy()
            row['target'] = example['target']
            row['timestamp'] = example['timestamp']
            df_data.append(row)
        df = pd.DataFrame(df_data)
        df = df.fillna(0)
        feature_cols = [col for col in df.columns if col not in ['target', 'timestamp']]
        self.feature_columns = feature_cols
        X = df[feature_cols].values
        y = df['target'].values
        return X, y
    async def _generate_alpha_prediction(self, features: Dict[str, float], signal_data: Dict) -> Optional[AlphaPrediction]:
        """Generate alpha prediction using model ensemble"""
        try:
            feature_vector = np.array([features.get(col, 0) for col in self.feature_columns]).reshape(1, -1)
            model_predictions = {}
            feature_contributions = {}
            for model_name, model in self.models.items():
                try:
                    X_scaled = self.scalers[model_name].transform(feature_vector)
                    prediction = model.predict(X_scaled)[0]
                    model_predictions[model_name] = prediction
                    if hasattr(model, 'feature_importances_'):
                        importance = dict(zip(self.feature_columns, model.feature_importances_))
                        feature_contributions[model_name] = importance
                except Exception as e:
                    logger.error(f"❌ Error in {model_name} prediction: {e}")
            if not model_predictions:
                return None
            weights = self._calculate_model_weights()
            ensemble_prediction = sum(
                pred * weights.get(model_name, 1.0) 
                for model_name, pred in model_predictions.items()
            ) / sum(weights.get(model_name, 1.0) for model_name in model_predictions.keys())
            pred_values = list(model_predictions.values())
            confidence = 1.0 - (np.std(pred_values) / (np.mean(np.abs(pred_values)) + 1e-6))
            confidence = max(0.0, min(1.0, confidence))
            avg_contributions = {}
            for feature in self.feature_columns:
                contributions = [
                    importance.get(feature, 0) 
                    for importance in feature_contributions.values()
                ]
                avg_contributions[feature] = np.mean(contributions) if contributions else 0
            prediction = AlphaPrediction(
                symbol=signal_data.get('symbol', 'BTCUSDT'),
                predicted_return=ensemble_prediction,
                confidence=confidence,
                prediction_horizon=self.prediction_horizon,
                contributing_features=avg_contributions,
                model_consensus=model_predictions,
                timestamp=time.time(),
                metadata={
                    'num_models': len(model_predictions),
                    'feature_vector_size': len(self.feature_columns),
                    'signal_strength': signal_data.get('signal_strength', 0)
                }
            )
            return prediction
        except Exception as e:
            logger.error(f"❌ Error generating alpha prediction: {e}")
            return None
    def _calculate_model_weights(self) -> Dict[str, float]:
        """Calculate weights for ensemble based on model performance"""
        weights = {}
        for model_name, metrics in self.performance_metrics.items():
            r2_weight = max(0, metrics.r2)
            cv_weight = max(0, 1.0 - metrics.cross_val_score)
            weights[model_name] = (r2_weight + cv_weight) / 2
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            weights = {model_name: 1.0 for model_name in self.models.keys()}
        return weights
    async def _emit_alpha_prediction(self, prediction: AlphaPrediction):
        """Emit alpha prediction to handlers"""
        logger.info(f"🎯 ALPHA PREDICTION: {prediction.symbol} - "
                   f"Return: {prediction.predicted_return:.4f} "
                   f"(Confidence: {prediction.confidence:.2f})")
        self.prediction_history.append(prediction)
        for handler in self.alpha_handlers:
            try:
                await asyncio.create_task(handler(prediction))
            except Exception as e:
                logger.error(f"❌ Alpha handler error: {e}")
    async def _save_models(self):
        """Save trained models to disk"""
        try:
            for model_name, model in self.models.items():
                model_path = f"data/models/{model_name}_alpha_model.joblib"
                scaler_path = f"data/models/{model_name}_scaler.joblib"
                joblib.dump(model, model_path)
                joblib.dump(self.scalers[model_name], scaler_path)
            joblib.dump(self.feature_columns, "data/models/feature_columns.joblib")
            logger.info("💾 Models saved successfully")
        except Exception as e:
            logger.error(f"❌ Error saving models: {e}")
    def load_models(self):
        """Load trained models from disk"""
        try:
            for model_name in self.models.keys():
                model_path = f"data/models/{model_name}_alpha_model.joblib"
                scaler_path = f"data/models/{model_name}_scaler.joblib"
                if os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)
                    self.scalers[model_name] = joblib.load(scaler_path)
            feature_path = "data/models/feature_columns.joblib"
            if os.path.exists(feature_path):
                self.feature_columns = joblib.load(feature_path)
            logger.info("📂 Models loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
    def get_performance_summary(self) -> Dict:
        """Get performance summary of the alpha detection engine"""
        if not self.performance_metrics:
            return {"status": "No trained models"}
        summary = {
            "models_trained": len(self.performance_metrics),
            "training_samples": next(iter(self.performance_metrics.values())).training_samples,
            "best_model": max(self.performance_metrics.items(), key=lambda x: x[1].r2)[0],
            "model_performance": {
                name: {
                    "r2_score": metrics.r2,
                    "mse": metrics.mse,
                    "cross_val_score": metrics.cross_val_score
                }
                for name, metrics in self.performance_metrics.items()
            },
            "predictions_made": len(self.prediction_history),
            "last_retrain": self.last_retrain_time
        }
        return summary