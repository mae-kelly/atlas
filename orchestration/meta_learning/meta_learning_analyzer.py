import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from dataclasses import dataclass
import time

@dataclass
class LearningPattern:
    pattern_type: str
    description: str
    confidence: float
    frequency: int
    effectiveness: float
    conditions: Dict

class MetaLearningAnalyzer:
    """
    Analyzes patterns across all strategies to generate meta-learning insights
    """
    
    def __init__(self):
        self.pattern_database = []
        self.learning_rules = []
        
    def analyze_strategy_interactions(self, strategy_data: Dict) -> List[LearningPattern]:
        """
        Analyze how strategies interact and perform together
        """
        patterns = []
        
        try:
            # Pattern 1: Complementary strategies
            complementary_pairs = self._find_complementary_strategies(strategy_data)
            for pair, effectiveness in complementary_pairs.items():
                pattern = LearningPattern(
                    pattern_type="complementary_strategies",
                    description=f"Strategies {pair[0]} and {pair[1]} work well together",
                    confidence=0.7,
                    frequency=10,
                    effectiveness=effectiveness,
                    conditions={"strategy_pair": pair}
                )
                patterns.append(pattern)
            
            # Pattern 2: Regime-specific performance
            regime_patterns = self._analyze_regime_patterns(strategy_data)
            patterns.extend(regime_patterns)
            
            # Pattern 3: Timing patterns
            timing_patterns = self._analyze_timing_patterns(strategy_data)
            patterns.extend(timing_patterns)
            
        except Exception as e:
            print(f"❌ Strategy interaction analysis error: {e}")
        
        return patterns
    
    def _find_complementary_strategies(self, strategy_data: Dict) -> Dict[Tuple[str, str], float]:
        """Find strategy pairs that work well together"""
        complementary_pairs = {}
        
        try:
            strategy_ids = list(strategy_data.keys())
            
            for i in range(len(strategy_ids)):
                for j in range(i + 1, len(strategy_ids)):
                    strategy1 = strategy_ids[i]
                    strategy2 = strategy_ids[j]
                    
                    # Simple complementarity metric
                    perf1 = strategy_data[strategy1].get('recent_performance', 0)
                    perf2 = strategy_data[strategy2].get('recent_performance', 0)
                    
                    # If both perform well, they might be complementary
                    if perf1 > 0.1 and perf2 > 0.1:
                        effectiveness = (perf1 + perf2) / 2
                        complementary_pairs[(strategy1, strategy2)] = effectiveness
            
        except Exception as e:
            print(f"❌ Complementary strategy analysis error: {e}")
        
        return complementary_pairs
    
    def _analyze_regime_patterns(self, strategy_data: Dict) -> List[LearningPattern]:
        """Analyze regime-specific performance patterns"""
        patterns = []
        
        try:
            # Group strategies by regime performance
            regime_specialists = defaultdict(list)
            
            for strategy_id, data in strategy_data.items():
                regime_performance = data.get('regime_performance', {})
                
                for regime, performance in regime_performance.items():
                    if performance > 0.15:  # Strong performance in regime
                        regime_specialists[regime].append((strategy_id, performance))
            
            # Create patterns for regime specialists
            for regime, specialists in regime_specialists.items():
                if len(specialists) >= 2:  # Multiple specialists
                    specialists.sort(key=lambda x: x[1], reverse=True)
                    top_specialists = specialists[:3]  # Top 3
                    
                    pattern = LearningPattern(
                        pattern_type="regime_specialists",
                        description=f"Best strategies for {regime} market conditions",
                        confidence=0.8,
                        frequency=5,
                        effectiveness=np.mean([s[1] for s in top_specialists]),
                        conditions={"regime": regime, "specialists": [s[0] for s in top_specialists]}
                    )
                    patterns.append(pattern)
        
        except Exception as e:
            print(f"❌ Regime pattern analysis error: {e}")
        
        return patterns
    
    def _analyze_timing_patterns(self, strategy_data: Dict) -> List[LearningPattern]:
        """Analyze timing-based patterns"""
        patterns = []
        
        try:
            # Simple timing pattern: strategies that work well at different times
            # This is a placeholder for more sophisticated timing analysis
            
            pattern = LearningPattern(
                pattern_type="timing_optimization",
                description="Optimal timing patterns for strategy execution",
                confidence=0.6,
                frequency=3,
                effectiveness=0.1,
                conditions={"timing_window": "market_hours"}
            )
            patterns.append(pattern)
            
        except Exception as e:
            print(f"❌ Timing pattern analysis error: {e}")
        
        return patterns
    
    def generate_optimization_recommendations(self, patterns: List[LearningPattern]) -> List[Dict]:
        """
        Generate actionable optimization recommendations from learned patterns
        """
        recommendations = []
        
        try:
            for pattern in patterns:
                if pattern.confidence > 0.7 and pattern.effectiveness > 0.1:
                    
                    if pattern.pattern_type == "complementary_strategies":
                        recommendation = {
                            'type': 'increase_allocation',
                            'target': pattern.conditions['strategy_pair'],
                            'reason': pattern.description,
                            'confidence': pattern.confidence,
                            'expected_improvement': pattern.effectiveness * 0.5
                        }
                        recommendations.append(recommendation)
                    
                    elif pattern.pattern_type == "regime_specialists":
                        recommendation = {
                            'type': 'regime_based_allocation',
                            'target': pattern.conditions['specialists'],
                            'regime': pattern.conditions['regime'],
                            'reason': pattern.description,
                            'confidence': pattern.confidence,
                            'expected_improvement': pattern.effectiveness * 0.3
                        }
                        recommendations.append(recommendation)
        
        except Exception as e:
            print(f"❌ Recommendation generation error: {e}")
        
        return recommendations
