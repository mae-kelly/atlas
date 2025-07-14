import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import time
from loguru import logger
from collections import deque, defaultdict
import cvxpy as cp
from scipy.optimize import minimize
from scipy import stats
import empyrical

class RiskLevel(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"

@dataclass
class PositionSizeRecommendation:
    symbol: str
    recommended_size: float  # Position size as % of portfolio
    max_position_size: float  # Maximum allowed size
    kelly_fraction: float    # Kelly criterion optimal fraction
    risk_adjusted_size: float  # Size after risk adjustments
    confidence_adjustment: float  # Adjustment based on prediction confidence
    portfolio_heat: float    # Current portfolio risk level
    expected_return: float   # Expected return from alpha prediction
    expected_volatility: float  # Expected position volatility
    sharpe_estimate: float   # Estimated Sharpe ratio
    risk_metrics: Dict       # Additional risk metrics
    timestamp: float

@dataclass
class RiskMetrics:
    portfolio_var: float     # Value at Risk
    portfolio_cvar: float    # Conditional Value at Risk
    max_drawdown: float      # Maximum drawdown
    volatility: float        # Portfolio volatility
    correlation_risk: float  # Concentration risk from correlations
    leverage: float          # Current leverage
    liquidity_risk: float    # Liquidity risk score
    regime_risk: float       # Market regime risk

class RiskManagementEngine:
    """
    Advanced risk management system that determines optimal position sizes
    and monitors portfolio risk in real-time
    """
    
    def __init__(self, 
                 base_capital: float = 100000.0,
                 max_portfolio_risk: float = 0.02,  # 2% daily VaR
                 max_single_position: float = 0.10,  # 10% max per position
                 max_sector_exposure: float = 0.30,   # 30% max per sector
                 target_sharpe: float = 1.5):
        
        self.base_capital = base_capital
        self.current_capital = base_capital
        self.max_portfolio_risk = max_portfolio_risk
        self.max_single_position = max_single_position
        self.max_sector_exposure = max_sector_exposure
        self.target_sharpe = target_sharpe
        
        # Portfolio tracking
        self.positions = {}  # Current positions
        self.position_history = deque(maxlen=1000)
        self.returns_history = deque(maxlen=252)  # ~1 year daily returns
        self.pnl_history = deque(maxlen=1000)
        
        # Risk models
        self.volatility_models = {}
        self.correlation_matrix = None
        self.var_models = {}
        
        # Market regime detection
        self.market_regime = "normal"
        self.regime_confidence = 0.5
        
        # Risk callbacks
        self.risk_alert_handlers = []
        
        # Performance tracking
        self.performance_metrics = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'avg_holding_period': 0.0
        }
        
        logger.info("🛡️ Risk Management Engine initialized")
    
    def add_risk_alert_handler(self, handler):
        """Add callback for risk alerts"""
        self.risk_alert_handlers.append(handler)
    
    async def evaluate_position_size(self, alpha_prediction) -> PositionSizeRecommendation:
        """
        Determine optimal position size for an alpha prediction
        """
        try:
            symbol = alpha_prediction.symbol
            expected_return = alpha_prediction.predicted_return
            confidence = alpha_prediction.confidence
            
            # 1. Calculate base position size using Kelly Criterion
            kelly_fraction = self._calculate_kelly_fraction(
                expected_return, 
                alpha_prediction, 
                symbol
            )
            
            # 2. Estimate position volatility
            estimated_volatility = self._estimate_position_volatility(symbol, alpha_prediction)
            
            # 3. Calculate risk-adjusted position size
            risk_adjusted_size = self._calculate_risk_adjusted_size(
                kelly_fraction, 
                estimated_volatility, 
                symbol
            )
            
            # 4. Apply confidence adjustments
            confidence_adjusted_size = risk_adjusted_size * self._confidence_multiplier(confidence)
            
            # 5. Apply portfolio-level constraints
            portfolio_adjusted_size = await self._apply_portfolio_constraints(
                confidence_adjusted_size, 
                symbol
            )
            
            # 6. Calculate final recommendation
            final_size = min(
                portfolio_adjusted_size,
                self.max_single_position,
                self._calculate_max_size_by_liquidity(symbol)
            )
            
            # 7. Generate risk metrics
            risk_metrics = await self._calculate_position_risk_metrics(
                symbol, 
                final_size, 
                estimated_volatility
            )
            
            # 8. Calculate expected Sharpe ratio
            expected_sharpe = expected_return / estimated_volatility if estimated_volatility > 0 else 0
            
            recommendation = PositionSizeRecommendation(
                symbol=symbol,
                recommended_size=final_size,
                max_position_size=self.max_single_position,
                kelly_fraction=kelly_fraction,
                risk_adjusted_size=risk_adjusted_size,
                confidence_adjustment=self._confidence_multiplier(confidence),
                portfolio_heat=await self._calculate_portfolio_heat(),
                expected_return=expected_return,
                expected_volatility=estimated_volatility,
                sharpe_estimate=expected_sharpe,
                risk_metrics=risk_metrics,
                timestamp=time.time()
            )
            
            # Log recommendation
            await self._log_position_recommendation(recommendation)
            
            return recommendation
            
        except Exception as e:
            logger.error(f"❌ Error calculating position size: {e}")
            return self._create_minimal_position_recommendation(alpha_prediction.symbol)
    
    def _calculate_kelly_fraction(self, expected_return: float, alpha_prediction, symbol: str) -> float:
        """
        Calculate Kelly Criterion optimal fraction
        Kelly% = (bp - q) / b
        where b = odds, p = win probability, q = loss probability
        """
        try:
            # Estimate win probability from prediction confidence and historical data
            base_win_prob = 0.5 + (alpha_prediction.confidence - 0.5) * 0.4  # Scale confidence to win prob
            
            # Adjust based on historical performance for this symbol
            historical_adjustment = self._get_historical_win_rate_adjustment(symbol)
            win_prob = min(0.9, max(0.1, base_win_prob * historical_adjustment))
            
            # Estimate average win/loss ratio
            avg_win_ratio = self._estimate_win_loss_ratio(symbol, expected_return)
            
            # Calculate Kelly fraction
            kelly = (win_prob * avg_win_ratio - (1 - win_prob)) / avg_win_ratio
            
            # Apply Kelly scaling (use fractional Kelly for safety)
            kelly_fraction = max(0, min(0.25, kelly * 0.25))  # Max 25% of Kelly, capped at 25%
            
            return kelly_fraction
            
        except Exception as e:
            logger.error(f"❌ Kelly calculation error: {e}")
            return 0.01  # Conservative fallback
    
    def _estimate_position_volatility(self, symbol: str, alpha_prediction) -> float:
        """
        Estimate volatility for position sizing
        """
        try:
            # Base volatility from historical data
            base_vol = self._get_historical_volatility(symbol)
            
            # Volatility from prediction metadata
            prediction_vol = alpha_prediction.metadata.get('volatility_estimate', base_vol)
            
            # Market regime adjustment
            regime_multiplier = self._get_regime_volatility_multiplier()
            
            # Time horizon adjustment
            horizon_days = alpha_prediction.prediction_horizon / (24 * 60)  # Convert minutes to days
            time_adjusted_vol = prediction_vol * np.sqrt(horizon_days)
            
            estimated_vol = time_adjusted_vol * regime_multiplier
            
            # Ensure reasonable bounds
            return max(0.01, min(0.50, estimated_vol))  # Between 1% and 50%
            
        except Exception as e:
            logger.error(f"❌ Volatility estimation error: {e}")
            return 0.02  # 2% default
    
    def _calculate_risk_adjusted_size(self, kelly_fraction: float, volatility: float, symbol: str) -> float:
        """
        Adjust position size based on various risk factors
        """
        try:
            # Start with Kelly fraction
            base_size = kelly_fraction
            
            # Volatility adjustment (reduce size for high volatility)
            vol_adjustment = min(1.0, 0.02 / volatility) if volatility > 0 else 1.0
            
            # Correlation adjustment (reduce size if highly correlated with existing positions)
            correlation_adjustment = self._calculate_correlation_adjustment(symbol)
            
            # Liquidity adjustment
            liquidity_adjustment = self._calculate_liquidity_adjustment(symbol)
            
            # Market regime adjustment
            regime_adjustment = self._get_regime_size_adjustment()
            
            # Combine all adjustments
            risk_adjusted_size = (
                base_size * 
                vol_adjustment * 
                correlation_adjustment * 
                liquidity_adjustment * 
                regime_adjustment
            )
            
            return max(0.001, risk_adjusted_size)  # Minimum 0.1% position
            
        except Exception as e:
            logger.error(f"❌ Risk adjustment error: {e}")
            return kelly_fraction * 0.5  # Conservative fallback
    
    def _confidence_multiplier(self, confidence: float) -> float:
        """
        Adjust position size based on prediction confidence
        """
        # S-curve scaling for confidence
        # High confidence gets exponential boost, low confidence gets heavy penalty
        if confidence < 0.3:
            return 0.1  # Very conservative for low confidence
        elif confidence < 0.5:
            return 0.3
        elif confidence < 0.7:
            return 0.6
        elif confidence < 0.8:
            return 0.8
        elif confidence < 0.9:
            return 1.0
        else:
            return 1.2  # Small boost for very high confidence
    
    async def _apply_portfolio_constraints(self, position_size: float, symbol: str) -> float:
        """
        Apply portfolio-level constraints
        """
        try:
            # Check current portfolio heat
            current_heat = await self._calculate_portfolio_heat()
            
            # Reduce size if portfolio is already hot
            if current_heat > 0.8:
                heat_adjustment = 0.5
            elif current_heat > 0.6:
                heat_adjustment = 0.7
            elif current_heat > 0.4:
                heat_adjustment = 0.9
            else:
                heat_adjustment = 1.0
            
            # Check sector concentration
            sector_adjustment = self._calculate_sector_concentration_adjustment(symbol)
            
            # Apply adjustments
            constrained_size = position_size * heat_adjustment * sector_adjustment
            
            return constrained_size
            
        except Exception as e:
            logger.error(f"❌ Portfolio constraint error: {e}")
            return position_size * 0.5
    
    async def _calculate_portfolio_heat(self) -> float:
        """
        Calculate current portfolio risk level (0-1 scale)
        """
        try:
            if not self.positions:
                return 0.0
            
            # Calculate position-level risk
            total_position_risk = sum(
                abs(pos['size']) * pos.get('volatility', 0.02)
                for pos in self.positions.values()
            )
            
            # Calculate correlation-adjusted risk
            if len(self.positions) > 1:
                correlation_risk = self._calculate_portfolio_correlation_risk()
            else:
                correlation_risk = total_position_risk
            
            # Normalize to 0-1 scale
            heat = min(1.0, correlation_risk / self.max_portfolio_risk)
            
            return heat
            
        except Exception as e:
            logger.error(f"❌ Portfolio heat calculation error: {e}")
            return 0.5  # Moderate heat assumption
    
    def _calculate_portfolio_correlation_risk(self) -> float:
        """
        Calculate portfolio risk accounting for correlations
        """
        try:
            if len(self.positions) < 2:
                return sum(abs(pos['size']) * pos.get('volatility', 0.02) for pos in self.positions.values())
            
            symbols = list(self.positions.keys())
            weights = np.array([self.positions[s]['size'] for s in symbols])
            volatilities = np.array([self.positions[s].get('volatility', 0.02) for s in symbols])
            
            # Use estimated correlation matrix
            corr_matrix = self._get_correlation_matrix(symbols)
            
            # Calculate portfolio variance
            cov_matrix = np.outer(volatilities, volatilities) * corr_matrix
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_risk = np.sqrt(portfolio_variance)
            
            return portfolio_risk
            
        except Exception as e:
            logger.error(f"❌ Correlation risk calculation error: {e}")
            return sum(abs(pos['size']) * pos.get('volatility', 0.02) for pos in self.positions.values())
    
    def _get_correlation_matrix(self, symbols: List[str]) -> np.ndarray:
        """
        Get or estimate correlation matrix for symbols
        """
        n = len(symbols)
        
        # Start with identity matrix
        corr_matrix = np.eye(n)
        
        # Add estimated correlations (in real implementation, use historical data)
        for i in range(n):
            for j in range(i+1, n):
                # Estimate correlation based on symbol similarity
                corr = self._estimate_correlation(symbols[i], symbols[j])
                corr_matrix[i, j] = corr_matrix[j, i] = corr
        
        return corr_matrix
    
    def _estimate_correlation(self, symbol1: str, symbol2: str) -> float:
        """
        Estimate correlation between two symbols
        """
        # Simple heuristic-based correlation estimation
        # In production, use historical price data
        
        if symbol1 == symbol2:
            return 1.0
        
        # Crypto pairs tend to be highly correlated
        if 'USDT' in symbol1 and 'USDT' in symbol2:
            return 0.7  # High crypto correlation
        
        # Different asset classes
        return 0.3  # Moderate default correlation
    
    async def _calculate_position_risk_metrics(self, symbol: str, position_size: float, volatility: float) -> Dict:
        """
        Calculate comprehensive risk metrics for a position
        """
        try:
            position_value = self.current_capital * position_size
            
            # Value at Risk (1-day, 95% confidence)
            var_95 = position_value * volatility * stats.norm.ppf(0.05)  # 5th percentile
            
            # Conditional Value at Risk (Expected Shortfall)
            cvar_95 = position_value * volatility * stats.norm.pdf(stats.norm.ppf(0.05)) / 0.05
            
            # Maximum theoretical loss
            max_loss = position_value  # 100% loss in extreme scenario
            
            # Liquidity risk score
            liquidity_score = self._calculate_liquidity_risk_score(symbol)
            
            # Time to liquidate estimate
            time_to_liquidate = self._estimate_liquidation_time(symbol, position_size)
            
            return {
                'position_value': position_value,
                'daily_var_95': abs(var_95),
                'daily_cvar_95': abs(cvar_95),
                'max_theoretical_loss': max_loss,
                'liquidity_risk_score': liquidity_score,
                'estimated_liquidation_time_hours': time_to_liquidate,
                'risk_reward_ratio': abs(var_95) / position_value if position_value > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Risk metrics calculation error: {e}")
            return {'error': 'Unable to calculate risk metrics'}
    
    async def _log_position_recommendation(self, recommendation: PositionSizeRecommendation):
        """
        Log position size recommendation
        """
        direction = "📈" if recommendation.expected_return > 0 else "📉"
        
        logger.info(f"{direction} POSITION SIZING: {recommendation.symbol}")
        logger.info(f"   Recommended Size: {recommendation.recommended_size:.2%}")
        logger.info(f"   Kelly Fraction: {recommendation.kelly_fraction:.2%}")
        logger.info(f"   Expected Return: {recommendation.expected_return:.2%}")
        logger.info(f"   Expected Vol: {recommendation.expected_volatility:.2%}")
        logger.info(f"   Sharpe Estimate: {recommendation.sharpe_estimate:.2f}")
        logger.info(f"   Portfolio Heat: {recommendation.portfolio_heat:.2%}")
    
    def update_position(self, symbol: str, size: float, entry_price: float):
        """
        Update position tracking
        """
        self.positions[symbol] = {
            'size': size,
            'entry_price': entry_price,
            'entry_time': time.time(),
            'volatility': self._get_historical_volatility(symbol),
            'unrealized_pnl': 0.0
        }
        
        # Log position update
        logger.info(f"📊 Position Updated: {symbol} - Size: {size:.2%} @ ${entry_price}")
    
    def update_position_pnl(self, symbol: str, current_price: float):
        """
        Update position P&L
        """
        if symbol in self.positions:
            position = self.positions[symbol]
            entry_price = position['entry_price']
            size = position['size']
            
            # Calculate unrealized P&L
            price_change = (current_price - entry_price) / entry_price
            unrealized_pnl = self.current_capital * size * price_change
            
            position['unrealized_pnl'] = unrealized_pnl
            position['current_price'] = current_price
            
            # Update portfolio P&L history
            total_unrealized = sum(pos['unrealized_pnl'] for pos in self.positions.values())
            self.pnl_history.append({
                'timestamp': time.time(),
                'total_unrealized_pnl': total_unrealized,
                'total_positions': len(self.positions)
            })
    
    # Helper methods with placeholder implementations
    def _get_historical_volatility(self, symbol: str) -> float:
        """Get historical volatility for symbol"""
        # Placeholder - in production, calculate from historical data
        volatility_map = {
            'BTCUSDT': 0.04,  # 4% daily volatility
            'ETHUSDT': 0.05,  # 5% daily volatility
        }
        return volatility_map.get(symbol, 0.03)  # 3% default
    
    def _get_historical_win_rate_adjustment(self, symbol: str) -> float:
        """Adjust win probability based on historical performance"""
        # Placeholder - analyze historical prediction accuracy
        return 1.0  # No adjustment for now
    
    def _estimate_win_loss_ratio(self, symbol: str, expected_return: float) -> float:
        """Estimate average win/loss ratio"""
        # Placeholder - use historical data analysis
        return abs(expected_return) / 0.01 if expected_return != 0 else 2.0
    
    def _get_regime_volatility_multiplier(self) -> float:
        """Get volatility multiplier based on market regime"""
        regime_multipliers = {
            'bull': 0.8,
            'bear': 1.5,
            'sideways': 1.0,
            'crisis': 2.0
        }
        return regime_multipliers.get(self.market_regime, 1.0)
    
    def _calculate_correlation_adjustment(self, symbol: str) -> float:
        """Reduce size if highly correlated with existing positions"""
        if not self.positions:
            return 1.0
        
        # Calculate average correlation with existing positions
        correlations = [self._estimate_correlation(symbol, existing) for existing in self.positions.keys()]
        avg_correlation = np.mean(correlations) if correlations else 0
        
        # Reduce size for high correlation
        return max(0.5, 1.0 - abs(avg_correlation) * 0.5)
    
    def _calculate_liquidity_adjustment(self, symbol: str) -> float:
        """Adjust size based on liquidity"""
        # Placeholder - in production, use market depth data
        return 1.0
    
    def _get_regime_size_adjustment(self) -> float:
        """Adjust position sizes based on market regime"""
        regime_adjustments = {
            'bull': 1.2,
            'bear': 0.7,
            'sideways': 1.0,
            'crisis': 0.3
        }
        return regime_adjustments.get(self.market_regime, 1.0)
    
    def _calculate_max_size_by_liquidity(self, symbol: str) -> float:
        """Calculate maximum position size based on liquidity"""
        # Placeholder - use market depth analysis
        return self.max_single_position
    
    def _calculate_sector_concentration_adjustment(self, symbol: str) -> float:
        """Adjust for sector concentration"""
        # Placeholder - implement sector classification
        return 1.0
    
    def _calculate_liquidity_risk_score(self, symbol: str) -> float:
        """Calculate liquidity risk score (0-1)"""
        # Placeholder - analyze market depth, spreads
        return 0.2  # Low risk default
    
    def _estimate_liquidation_time(self, symbol: str, position_size: float) -> float:
        """Estimate time to liquidate position (hours)"""
        # Placeholder - based on volume analysis
        return 0.5  # 30 minutes default
    
    def _create_minimal_position_recommendation(self, symbol: str) -> PositionSizeRecommendation:
        """Create minimal safe position recommendation"""
        return PositionSizeRecommendation(
            symbol=symbol,
            recommended_size=0.001,  # 0.1% minimal position
            max_position_size=self.max_single_position,
            kelly_fraction=0.001,
            risk_adjusted_size=0.001,
            confidence_adjustment=0.1,
            portfolio_heat=0.0,
            expected_return=0.0,
            expected_volatility=0.02,
            sharpe_estimate=0.0,
            risk_metrics={},
            timestamp=time.time()
        )
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio risk summary"""
        try:
            total_exposure = sum(abs(pos['size']) for pos in self.positions.values())
            total_unrealized = sum(pos.get('unrealized_pnl', 0) for pos in self.positions.values())
            
            return {
                'total_capital': self.current_capital,
                'total_exposure': total_exposure,
                'total_unrealized_pnl': total_unrealized,
                'number_of_positions': len(self.positions),
                'portfolio_heat': asyncio.run(self._calculate_portfolio_heat()),
                'max_single_position': self.max_single_position,
                'current_positions': {
                    symbol: {
                        'size': pos['size'],
                        'unrealized_pnl': pos.get('unrealized_pnl', 0),
                        'entry_price': pos['entry_price']
                    }
                    for symbol, pos in self.positions.items()
                },
                'risk_utilization': total_exposure / 1.0,  # Risk budget utilization
                'market_regime': self.market_regime
            }
        except Exception as e:
            logger.error(f"❌ Portfolio summary error: {e}")
            return {'error': 'Unable to generate portfolio summary'}
