import asyncio
import time
import json
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from loguru import logger
from enum import Enum
import uuid

class StrategyType(Enum):
    PRICE_MOMENTUM = "price_momentum"
    SENTIMENT_ALPHA = "sentiment_alpha"
    FUSION_SIGNALS = "fusion_signals"
    ML_ALPHA = "ml_alpha"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"

@dataclass
class StrategySignal:
    strategy_id: str
    strategy_type: StrategyType
    symbol: str
    signal_strength: float  # 0-1
    expected_return: float
    confidence: float
    holding_period: int  # minutes
    metadata: Dict
    timestamp: float

@dataclass
class StrategyPerformance:
    strategy_id: str
    strategy_type: StrategyType
    total_signals: int
    successful_signals: int
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_holding_period: float
    recent_performance: float  # Last 30 days
    regime_performance: Dict[MarketRegime, float]
    capital_allocated: float
    risk_adjusted_return: float

@dataclass
class MetaLearningInsight:
    insight_type: str
    description: str
    confidence: float
    applicable_regimes: List[MarketRegime]
    strategy_weights: Dict[str, float]
    expected_improvement: float
    timestamp: float

class StrategyOrchestrationEngine:
    """
    Master orchestrator that coordinates all trading strategies and learns
    which approaches work best in different market conditions
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 max_strategies: int = 10,
                 rebalance_frequency: int = 3600):  # 1 hour
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_strategies = max_strategies
        self.rebalance_frequency = rebalance_frequency
        
        # Strategy management
        self.active_strategies: Dict[str, Dict] = {}
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self.strategy_signals: deque = deque(maxlen=10000)
        
        # Capital allocation
        self.strategy_allocations: Dict[str, float] = {}
        self.allocation_history: deque = deque(maxlen=1000)
        
        # Market regime detection
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_confidence = 0.5
        self.regime_history: deque = deque(maxlen=1000)
        
        # Meta-learning
        self.meta_insights: List[MetaLearningInsight] = []
        self.learning_history: deque = deque(maxlen=5000)
        self.strategy_correlations: Dict[Tuple[str, str], float] = {}
        
        # Performance tracking
        self.orchestration_metrics = {
            'total_strategies_managed': 0,
            'total_signals_processed': 0,
            'total_capital_deployed': 0,
            'meta_learning_iterations': 0,
            'regime_transitions': 0,
            'optimization_cycles': 0
        }
        
        # Callbacks
        self.signal_handlers: List[Callable] = []
        self.allocation_handlers: List[Callable] = []
        self.insight_handlers: List[Callable] = []
        
        # Timing
        self.last_rebalance_time = time.time()
        self.last_meta_learning_time = time.time()
        
        logger.info("🧠 Strategy Orchestration Engine initialized")
    
    def add_signal_handler(self, handler: Callable):
        """Add handler for strategy signals"""
        self.signal_handlers.append(handler)
    
    def add_allocation_handler(self, handler: Callable):
        """Add handler for capital allocation updates"""
        self.allocation_handlers.append(handler)
    
    def add_insight_handler(self, handler: Callable):
        """Add handler for meta-learning insights"""
        self.insight_handlers.append(handler)
    
    async def register_strategy(self, strategy_id: str, strategy_type: StrategyType, 
                              strategy_config: Dict) -> bool:
        """
        Register a new trading strategy with the orchestrator
        """
        try:
            if len(self.active_strategies) >= self.max_strategies:
                logger.warning(f"⚠️ Maximum strategies ({self.max_strategies}) reached")
                return False
            
            if strategy_id in self.active_strategies:
                logger.warning(f"⚠️ Strategy {strategy_id} already registered")
                return False
            
            # Register strategy
            self.active_strategies[strategy_id] = {
                'type': strategy_type,
                'config': strategy_config,
                'registration_time': time.time(),
                'status': 'active',
                'signals_generated': 0,
                'last_signal_time': 0
            }
            
            # Initialize performance tracking
            self.strategy_performance[strategy_id] = StrategyPerformance(
                strategy_id=strategy_id,
                strategy_type=strategy_type,
                total_signals=0,
                successful_signals=0,
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                avg_holding_period=0.0,
                recent_performance=0.0,
                regime_performance={regime: 0.0 for regime in MarketRegime},
                capital_allocated=0.0,
                risk_adjusted_return=0.0
            )
            
            # Initial capital allocation
            await self._allocate_capital_to_strategy(strategy_id, strategy_type)
            
            self.orchestration_metrics['total_strategies_managed'] += 1
            
            logger.info(f"✅ Strategy registered: {strategy_id} ({strategy_type.value})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Strategy registration error: {e}")
            return False
    
    async def process_strategy_signal(self, signal: StrategySignal) -> bool:
        """
        Process incoming signal from a strategy
        """
        try:
            # Validate signal
            if signal.strategy_id not in self.active_strategies:
                logger.warning(f"⚠️ Signal from unregistered strategy: {signal.strategy_id}")
                return False
            
            # Store signal
            self.strategy_signals.append(signal)
            
            # Update strategy metrics
            strategy = self.active_strategies[signal.strategy_id]
            strategy['signals_generated'] += 1
            strategy['last_signal_time'] = signal.timestamp
            
            # Update performance tracking
            performance = self.strategy_performance[signal.strategy_id]
            performance.total_signals += 1
            
            # Evaluate signal quality and context
            signal_quality = await self._evaluate_signal_quality(signal)
            
            # Adjust signal based on current regime and meta-learning insights
            adjusted_signal = await self._apply_meta_learning_adjustments(signal, signal_quality)
            
            # Forward to execution if signal passes filters
            if await self._should_execute_signal(adjusted_signal):
                await self._emit_execution_signal(adjusted_signal)
            
            # Update learning systems
            await self._update_meta_learning(signal, signal_quality)
            
            self.orchestration_metrics['total_signals_processed'] += 1
            
            # Check if rebalancing is needed
            if self._should_rebalance():
                await self._rebalance_capital_allocation()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Signal processing error: {e}")
            return False
    
    async def _evaluate_signal_quality(self, signal: StrategySignal) -> Dict:
        """
        Evaluate the quality and context of a strategy signal
        """
        try:
            # Base quality from signal strength and confidence
            base_quality = (signal.signal_strength + signal.confidence) / 2
            
            # Regime appropriateness
            strategy_performance = self.strategy_performance[signal.strategy_id]
            regime_performance = strategy_performance.regime_performance.get(self.current_regime, 0.0)
            regime_adjustment = min(regime_performance / 0.1, 2.0) if regime_performance > 0 else 0.5
            
            # Recent strategy performance
            recent_performance = strategy_performance.recent_performance
            performance_adjustment = min(max(recent_performance + 1, 0.2), 2.0)
            
            # Timing quality (avoid signal clustering)
            timing_quality = self._calculate_timing_quality(signal)
            
            # Correlation with other active signals
            correlation_penalty = self._calculate_correlation_penalty(signal)
            
            # Final quality score
            quality_score = (
                base_quality * 
                regime_adjustment * 
                performance_adjustment * 
                timing_quality * 
                (1 - correlation_penalty)
            )
            
            return {
                'quality_score': min(max(quality_score, 0.0), 1.0),
                'base_quality': base_quality,
                'regime_adjustment': regime_adjustment,
                'performance_adjustment': performance_adjustment,
                'timing_quality': timing_quality,
                'correlation_penalty': correlation_penalty,
                'regime': self.current_regime.value,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Signal quality evaluation error: {e}")
            return {'quality_score': 0.5}
    
    def _calculate_timing_quality(self, signal: StrategySignal) -> float:
        """
        Calculate timing quality to avoid signal clustering
        """
        try:
            current_time = signal.timestamp
            recent_signals = [
                s for s in self.strategy_signals 
                if s.strategy_id == signal.strategy_id and 
                current_time - s.timestamp < 3600  # Last hour
            ]
            
            if len(recent_signals) <= 1:
                return 1.0
            
            # Penalize excessive signal frequency
            time_gaps = []
            for i in range(1, len(recent_signals)):
                gap = recent_signals[i].timestamp - recent_signals[i-1].timestamp
                time_gaps.append(gap)
            
            avg_gap = np.mean(time_gaps) if time_gaps else 3600
            optimal_gap = 1800  # 30 minutes
            
            timing_quality = min(avg_gap / optimal_gap, 1.0)
            return max(timing_quality, 0.2)  # Minimum 0.2
            
        except Exception as e:
            logger.error(f"❌ Timing quality calculation error: {e}")
            return 1.0
    
    def _calculate_correlation_penalty(self, signal: StrategySignal) -> float:
        """
        Calculate penalty for correlated signals
        """
        try:
            current_time = signal.timestamp
            recent_signals = [
                s for s in self.strategy_signals 
                if current_time - s.timestamp < 1800 and  # Last 30 minutes
                s.symbol == signal.symbol and 
                s.strategy_id != signal.strategy_id
            ]
            
            if not recent_signals:
                return 0.0
            
            # Count similar direction signals
            same_direction = sum(
                1 for s in recent_signals 
                if (s.expected_return > 0) == (signal.expected_return > 0)
            )
            
            # Penalty increases with number of correlated signals
            correlation_penalty = min(same_direction * 0.2, 0.8)
            return correlation_penalty
            
        except Exception as e:
            logger.error(f"❌ Correlation penalty calculation error: {e}")
            return 0.0
    
    async def _apply_meta_learning_adjustments(self, signal: StrategySignal, quality: Dict) -> StrategySignal:
        """
        Apply meta-learning insights to adjust signal parameters
        """
        try:
            # Find applicable insights
            applicable_insights = [
                insight for insight in self.meta_insights
                if self.current_regime in insight.applicable_regimes
            ]
            
            if not applicable_insights:
                return signal
            
            # Apply strategy weight adjustments
            strategy_weight_adjustment = 1.0
            for insight in applicable_insights:
                if signal.strategy_id in insight.strategy_weights:
                    weight = insight.strategy_weights[signal.strategy_id]
                    confidence = insight.confidence
                    strategy_weight_adjustment *= (1 + (weight - 1) * confidence)
            
            # Adjust signal strength
            adjusted_strength = signal.signal_strength * strategy_weight_adjustment
            adjusted_strength = min(max(adjusted_strength, 0.0), 1.0)
            
            # Create adjusted signal
            adjusted_signal = StrategySignal(
                strategy_id=signal.strategy_id,
                strategy_type=signal.strategy_type,
                symbol=signal.symbol,
                signal_strength=adjusted_strength,
                expected_return=signal.expected_return,
                confidence=signal.confidence,
                holding_period=signal.holding_period,
                metadata={
                    **signal.metadata,
                    'original_strength': signal.signal_strength,
                    'meta_adjustment': strategy_weight_adjustment,
                    'quality_score': quality['quality_score']
                },
                timestamp=signal.timestamp
            )
            
            return adjusted_signal
            
        except Exception as e:
            logger.error(f"❌ Meta-learning adjustment error: {e}")
            return signal
    
    async def _should_execute_signal(self, signal: StrategySignal) -> bool:
        """
        Determine if signal should be executed
        """
        try:
            # Minimum quality threshold
            quality_score = signal.metadata.get('quality_score', 0.5)
            if quality_score < 0.3:
                return False
            
            # Check capital allocation limits
            strategy_allocation = self.strategy_allocations.get(signal.strategy_id, 0.0)
            if strategy_allocation < 0.01:  # Less than 1% allocation
                return False
            
            # Check signal strength
            if signal.signal_strength < 0.2:
                return False
            
            # Check regime appropriateness
            strategy_performance = self.strategy_performance[signal.strategy_id]
            regime_performance = strategy_performance.regime_performance.get(self.current_regime, 0.0)
            if regime_performance < -0.1:  # Consistently bad in this regime
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Signal execution decision error: {e}")
            return False
    
    async def _emit_execution_signal(self, signal: StrategySignal):
        """
        Emit signal for execution
        """
        try:
            # Calculate position size based on strategy allocation
            strategy_allocation = self.strategy_allocations.get(signal.strategy_id, 0.0)
            allocated_capital = self.current_capital * strategy_allocation
            
            execution_signal = {
                'signal_id': str(uuid.uuid4()),
                'strategy_id': signal.strategy_id,
                'strategy_type': signal.strategy_type.value,
                'symbol': signal.symbol,
                'signal_strength': signal.signal_strength,
                'expected_return': signal.expected_return,
                'confidence': signal.confidence,
                'allocated_capital': allocated_capital,
                'holding_period': signal.holding_period,
                'execution_priority': signal.signal_strength * signal.confidence,
                'metadata': signal.metadata,
                'timestamp': time.time()
            }
            
            # Notify all handlers
            for handler in self.signal_handlers:
                try:
                    await asyncio.create_task(handler(execution_signal))
                except Exception as e:
                    logger.error(f"❌ Signal handler error: {e}")
            
            logger.info(f"🎯 Execution Signal: {signal.strategy_id} - {signal.symbol} "
                       f"(Strength: {signal.signal_strength:.2f}, Capital: ${allocated_capital:,.0f})")
            
        except Exception as e:
            logger.error(f"❌ Signal emission error: {e}")
    
    def _should_rebalance(self) -> bool:
        """Check if capital rebalancing is needed"""
        return time.time() - self.last_rebalance_time > self.rebalance_frequency
    
    async def _rebalance_capital_allocation(self):
        """
        Rebalance capital allocation based on strategy performance and meta-learning
        """
        try:
            logger.info("⚖️ Rebalancing capital allocation...")
            
            # Calculate new allocations based on performance
            new_allocations = await self._calculate_optimal_allocations()
            
            # Apply gradual rebalancing (don't change too quickly)
            smoothing_factor = 0.3  # 30% adjustment per rebalancing
            
            for strategy_id in self.active_strategies.keys():
                current_allocation = self.strategy_allocations.get(strategy_id, 0.0)
                target_allocation = new_allocations.get(strategy_id, 0.0)
                
                new_allocation = (
                    current_allocation * (1 - smoothing_factor) + 
                    target_allocation * smoothing_factor
                )
                
                self.strategy_allocations[strategy_id] = new_allocation
            
            # Normalize allocations to sum to 1
            total_allocation = sum(self.strategy_allocations.values())
            if total_allocation > 0:
                for strategy_id in self.strategy_allocations:
                    self.strategy_allocations[strategy_id] /= total_allocation
            
            # Store allocation history
            self.allocation_history.append({
                'timestamp': time.time(),
                'allocations': self.strategy_allocations.copy(),
                'regime': self.current_regime.value
            })
            
            # Notify handlers
            for handler in self.allocation_handlers:
                try:
                    await asyncio.create_task(handler(self.strategy_allocations.copy()))
                except Exception as e:
                    logger.error(f"❌ Allocation handler error: {e}")
            
            self.last_rebalance_time = time.time()
            self.orchestration_metrics['optimization_cycles'] += 1
            
            logger.info(f"✅ Capital rebalanced across {len(self.strategy_allocations)} strategies")
            
        except Exception as e:
            logger.error(f"❌ Capital rebalancing error: {e}")
    
    async def _calculate_optimal_allocations(self) -> Dict[str, float]:
        """
        Calculate optimal capital allocations using performance and regime data
        """
        try:
            if not self.strategy_performance:
                return {}
            
            allocations = {}
            
            # Score each strategy
            strategy_scores = {}
            for strategy_id, performance in self.strategy_performance.items():
                # Base score from risk-adjusted return
                base_score = performance.risk_adjusted_return
                
                # Recent performance weight
                recent_weight = 0.4
                recent_score = performance.recent_performance
                
                # Regime performance weight
                regime_weight = 0.3
                regime_score = performance.regime_performance.get(self.current_regime, 0.0)
                
                # Win rate weight
                winrate_weight = 0.2
                winrate_score = performance.win_rate - 0.5  # Center around 0.5
                
                # Sharpe ratio weight
                sharpe_weight = 0.1
                sharpe_score = min(performance.sharpe_ratio / 2.0, 1.0)  # Normalize
                
                # Combined score
                total_score = (
                    base_score +
                    recent_score * recent_weight +
                    regime_score * regime_weight +
                    winrate_score * winrate_weight +
                    sharpe_score * sharpe_weight
                )
                
                # Apply minimum and maximum bounds
                strategy_scores[strategy_id] = max(0.01, min(total_score, 2.0))
            
            # Convert scores to allocations
            total_score = sum(strategy_scores.values())
            if total_score > 0:
                for strategy_id, score in strategy_scores.items():
                    allocations[strategy_id] = score / total_score
            else:
                # Equal allocation if no performance data
                equal_allocation = 1.0 / len(self.strategy_performance)
                allocations = {sid: equal_allocation for sid in self.strategy_performance.keys()}
            
            # Apply constraints
            max_single_allocation = 0.4  # Maximum 40% to any single strategy
            min_allocation = 0.05  # Minimum 5% to keep strategies alive
            
            for strategy_id in allocations:
                allocations[strategy_id] = min(allocations[strategy_id], max_single_allocation)
                allocations[strategy_id] = max(allocations[strategy_id], min_allocation)
            
            # Renormalize
            total_allocation = sum(allocations.values())
            if total_allocation > 0:
                for strategy_id in allocations:
                    allocations[strategy_id] /= total_allocation
            
            return allocations
            
        except Exception as e:
            logger.error(f"❌ Optimal allocation calculation error: {e}")
            return {}
    
    async def _allocate_capital_to_strategy(self, strategy_id: str, strategy_type: StrategyType):
        """
        Initial capital allocation for new strategy
        """
        try:
            # Equal allocation initially, will be optimized over time
            if self.active_strategies:
                initial_allocation = 1.0 / len(self.active_strategies)
            else:
                initial_allocation = 1.0
            
            self.strategy_allocations[strategy_id] = initial_allocation
            
            # Renormalize all allocations
            total_allocation = sum(self.strategy_allocations.values())
            if total_allocation > 0:
                for sid in self.strategy_allocations:
                    self.strategy_allocations[sid] /= total_allocation
            
        except Exception as e:
            logger.error(f"❌ Capital allocation error: {e}")
    
    async def _update_meta_learning(self, signal: StrategySignal, quality: Dict):
        """
        Update meta-learning systems with new signal data
        """
        try:
            # Store learning data
            learning_data = {
                'signal': asdict(signal),
                'quality': quality,
                'regime': self.current_regime.value,
                'timestamp': time.time()
            }
            
            self.learning_history.append(learning_data)
            
            # Run meta-learning every hour
            if time.time() - self.last_meta_learning_time > 3600:
                await self._run_meta_learning_analysis()
                self.last_meta_learning_time = time.time()
            
        except Exception as e:
            logger.error(f"❌ Meta-learning update error: {e}")
    
    async def _run_meta_learning_analysis(self):
        """
        Run comprehensive meta-learning analysis
        """
        try:
            logger.info("🔬 Running meta-learning analysis...")
            
            if len(self.learning_history) < 100:  # Need minimum data
                return
            
            # Analyze strategy performance patterns
            await self._analyze_strategy_patterns()
            
            # Analyze regime transitions
            await self._analyze_regime_patterns()
            
            # Generate new insights
            await self._generate_meta_insights()
            
            self.orchestration_metrics['meta_learning_iterations'] += 1
            
            logger.info(f"✅ Meta-learning complete - {len(self.meta_insights)} insights generated")
            
        except Exception as e:
            logger.error(f"❌ Meta-learning analysis error: {e}")
    
    async def _analyze_strategy_patterns(self):
        """
        Analyze patterns in strategy performance
        """
        try:
            # Group learning data by strategy
            strategy_data = defaultdict(list)
            for data in self.learning_history:
                strategy_id = data['signal']['strategy_id']
                strategy_data[strategy_id].append(data)
            
            # Analyze each strategy's patterns
            for strategy_id, data_points in strategy_data.items():
                if len(data_points) < 20:
                    continue
                
                # Analyze quality score patterns
                quality_scores = [d['quality']['quality_score'] for d in data_points]
                avg_quality = np.mean(quality_scores)
                quality_trend = self._calculate_trend(quality_scores)
                
                # Update strategy performance
                if strategy_id in self.strategy_performance:
                    performance = self.strategy_performance[strategy_id]
                    performance.recent_performance = avg_quality - 0.5  # Center around 0
                    
                    # Update regime-specific performance
                    regime_quality = defaultdict(list)
                    for dp in data_points:
                        regime = MarketRegime(dp['regime'])
                        regime_quality[regime].append(dp['quality']['quality_score'])
                    
                    for regime, scores in regime_quality.items():
                        if len(scores) >= 5:  # Minimum samples for regime
                            performance.regime_performance[regime] = np.mean(scores) - 0.5
            
        except Exception as e:
            logger.error(f"❌ Strategy pattern analysis error: {e}")
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in values using simple linear regression"""
        try:
            if len(values) < 5:
                return 0.0
            
            x = np.arange(len(values))
            y = np.array(values)
            
            # Simple linear regression
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            
            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sum((x - x_mean) ** 2)
            
            if denominator == 0:
                return 0.0
            
            slope = numerator / denominator
            return slope
            
        except Exception as e:
            logger.error(f"❌ Trend calculation error: {e}")
            return 0.0
    
    async def _analyze_regime_patterns(self):
        """
        Analyze market regime transition patterns
        """
        try:
            # Update current regime based on recent data
            await self._detect_current_regime()
            
            # Analyze regime transition effects on strategies
            # This would involve more complex analysis of how strategies
            # perform during regime changes
            
        except Exception as e:
            logger.error(f"❌ Regime pattern analysis error: {e}")
    
    async def _detect_current_regime(self):
        """
        Detect current market regime
        """
        try:
            # Simplified regime detection based on recent signal patterns
            # In production, this would use sophisticated market data analysis
            
            recent_data = list(self.learning_history)[-100:]  # Last 100 signals
            if len(recent_data) < 50:
                return
            
            # Analyze signal characteristics
            signal_strengths = [d['signal']['signal_strength'] for d in recent_data]
            expected_returns = [d['signal']['expected_return'] for d in recent_data]
            
            avg_strength = np.mean(signal_strengths)
            avg_return = np.mean(expected_returns)
            return_volatility = np.std(expected_returns)
            
            # Simple regime classification
            if return_volatility > 0.05:
                new_regime = MarketRegime.HIGH_VOLATILITY
            elif avg_return > 0.02:
                new_regime = MarketRegime.BULL
            elif avg_return < -0.02:
                new_regime = MarketRegime.BEAR
            elif return_volatility < 0.01:
                new_regime = MarketRegime.LOW_VOLATILITY
            else:
                new_regime = MarketRegime.SIDEWAYS
            
            # Update regime if changed
            if new_regime != self.current_regime:
                logger.info(f"📊 Market regime change: {self.current_regime.value} → {new_regime.value}")
                self.current_regime = new_regime
                self.orchestration_metrics['regime_transitions'] += 1
                
                # Store regime history
                self.regime_history.append({
                    'regime': new_regime.value,
                    'timestamp': time.time(),
                    'confidence': 0.7  # Simplified confidence
                })
            
        except Exception as e:
            logger.error(f"❌ Regime detection error: {e}")
    
    async def _generate_meta_insights(self):
        """
        Generate meta-learning insights from analyzed patterns
        """
        try:
            # Insight 1: Strategy performance by regime
            regime_insights = await self._generate_regime_insights()
            
            # Insight 2: Strategy correlation patterns
            correlation_insights = await self._generate_correlation_insights()
            
            # Insight 3: Timing optimization insights
            timing_insights = await self._generate_timing_insights()
            
            # Combine and store insights
            new_insights = regime_insights + correlation_insights + timing_insights
            
            # Add new insights (keep only recent ones)
            self.meta_insights.extend(new_insights)
            
            # Remove old insights (keep last 50)
            self.meta_insights = self.meta_insights[-50:]
            
            # Notify handlers
            for insight in new_insights:
                for handler in self.insight_handlers:
                    try:
                        await asyncio.create_task(handler(insight))
                    except Exception as e:
                        logger.error(f"❌ Insight handler error: {e}")
            
        except Exception as e:
            logger.error(f"❌ Meta-insight generation error: {e}")
    
    async def _generate_regime_insights(self) -> List[MetaLearningInsight]:
        """Generate insights about strategy performance in different regimes"""
        insights = []
        
        try:
            for regime in MarketRegime:
                # Find best performing strategies in this regime
                regime_performance = {}
                for strategy_id, performance in self.strategy_performance.items():
                    regime_perf = performance.regime_performance.get(regime, 0.0)
                    if regime_perf > 0.1:  # Significantly positive
                        regime_performance[strategy_id] = regime_perf
                
                if regime_performance:
                    # Create insight for this regime
                    insight = MetaLearningInsight(
                        insight_type="regime_optimization",
                        description=f"Optimal strategy allocation for {regime.value} market",
                        confidence=0.7,
                        applicable_regimes=[regime],
                        strategy_weights=regime_performance,
                        expected_improvement=max(regime_performance.values()) * 0.5,
                        timestamp=time.time()
                    )
                    insights.append(insight)
            
        except Exception as e:
            logger.error(f"❌ Regime insight generation error: {e}")
        
        return insights
    
    async def _generate_correlation_insights(self) -> List[MetaLearningInsight]:
        """Generate insights about strategy correlations"""
        insights = []
        
        try:
            # Analyze which strategies work well together
            # This is a simplified version - production would be more sophisticated
            
            strategy_pairs = []
            strategy_ids = list(self.strategy_performance.keys())
            
            for i in range(len(strategy_ids)):
                for j in range(i + 1, len(strategy_ids)):
                    strategy_pairs.append((strategy_ids[i], strategy_ids[j]))
            
            for pair in strategy_pairs:
                # Simplified correlation analysis
                perf1 = self.strategy_performance[pair[0]].recent_performance
                perf2 = self.strategy_performance[pair[1]].recent_performance
                
                # If both strategies are performing well, suggest increased allocation
                if perf1 > 0.1 and perf2 > 0.1:
                    strategy_weights = {pair[0]: 1.2, pair[1]: 1.2}
                    
                    insight = MetaLearningInsight(
                        insight_type="strategy_synergy",
                        description=f"Synergistic performance between {pair[0]} and {pair[1]}",
                        confidence=0.6,
                        applicable_regimes=[self.current_regime],
                        strategy_weights=strategy_weights,
                        expected_improvement=0.15,
                        timestamp=time.time()
                    )
                    insights.append(insight)
            
        except Exception as e:
            logger.error(f"❌ Correlation insight generation error: {e}")
        
        return insights
    
    async def _generate_timing_insights(self) -> List[MetaLearningInsight]:
        """Generate insights about optimal timing"""
        insights = []
        
        try:
            # Analyze time-of-day performance patterns
            # This is simplified - production would analyze actual time patterns
            
            insight = MetaLearningInsight(
                insight_type="timing_optimization",
                description="Optimal timing patterns identified",
                confidence=0.5,
                applicable_regimes=[self.current_regime],
                strategy_weights={},  # No specific strategy weights
                expected_improvement=0.1,
                timestamp=time.time()
            )
            insights.append(insight)
            
        except Exception as e:
            logger.error(f"❌ Timing insight generation error: {e}")
        
        return insights
    
    def get_orchestration_summary(self) -> Dict:
        """Get comprehensive orchestration summary"""
        try:
            # Strategy summaries
            strategy_summaries = {}
            for strategy_id, performance in self.strategy_performance.items():
                strategy_summaries[strategy_id] = {
                    'type': performance.strategy_type.value,
                    'allocation': self.strategy_allocations.get(strategy_id, 0.0),
                    'recent_performance': performance.recent_performance,
                    'total_signals': performance.total_signals,
                    'win_rate': performance.win_rate,
                    'regime_performance': {k.value: v for k, v in performance.regime_performance.items()}
                }
            
            # Recent insights
            recent_insights = [
                {
                    'type': insight.insight_type,
                    'description': insight.description,
                    'confidence': insight.confidence,
                    'expected_improvement': insight.expected_improvement
                }
                for insight in self.meta_insights[-5:]  # Last 5 insights
            ]
            
            return {
                'timestamp': time.time(),
                'current_regime': self.current_regime.value,
                'regime_confidence': self.regime_confidence,
                'total_strategies': len(self.active_strategies),
                'active_allocations': self.strategy_allocations.copy(),
                'strategy_performance': strategy_summaries,
                'orchestration_metrics': self.orchestration_metrics.copy(),
                'recent_insights': recent_insights,
                'meta_learning_insights_count': len(self.meta_insights),
                'signals_processed_today': len([
                    s for s in self.strategy_signals 
                    if time.time() - s.timestamp < 86400
                ])
            }
            
        except Exception as e:
            logger.error(f"❌ Orchestration summary error: {e}")
            return {'error': 'Unable to generate summary'}
    
    def save_orchestration_state(self, filepath: str = None):
        """Save orchestration state to file"""
        try:
            if not filepath:
                filepath = f"data/strategy_performance/orchestration_state_{int(time.time())}.json"
            
            state = {
                'active_strategies': self.active_strategies,
                'strategy_allocations': self.strategy_allocations,
                'current_regime': self.current_regime.value,
                'meta_insights': [asdict(insight) for insight in self.meta_insights],
                'orchestration_metrics': self.orchestration_metrics,
                'allocation_history': list(self.allocation_history),
                'regime_history': list(self.regime_history)
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.info(f"💾 Orchestration state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Save orchestration state error: {e}")
