import asyncio
import time
import random
import numpy as np
from orchestration.strategy_orchestration_engine import (
    StrategyOrchestrationEngine, StrategySignal, StrategyType, MarketRegime
)
class MockStrategy:
    """Mock strategy for testing orchestration"""
    def __init__(self, strategy_id: str, strategy_type: StrategyType, orchestrator: StrategyOrchestrationEngine):
        self.strategy_id = strategy_id
        self.strategy_type = strategy_type
        self.orchestrator = orchestrator
        self.running = False
        self.base_performance = random.uniform(0.3, 0.8)
        self.regime_preferences = {
            MarketRegime.BULL: random.uniform(0.5, 1.0),
            MarketRegime.BEAR: random.uniform(0.2, 0.7),
            MarketRegime.SIDEWAYS: random.uniform(0.4, 0.8),
            MarketRegime.HIGH_VOLATILITY: random.uniform(0.3, 0.9),
            MarketRegime.LOW_VOLATILITY: random.uniform(0.4, 0.7),
            MarketRegime.CRISIS: random.uniform(0.1, 0.5)
        }
    async def start(self):
        """Start generating signals"""
        self.running = True
        config = {
            'base_performance': self.base_performance,
            'regime_preferences': {k.value: v for k, v in self.regime_preferences.items()}
        }
        await self.orchestrator.register_strategy(self.strategy_id, self.strategy_type, config)
        while self.running:
            await self._generate_signal()
            await asyncio.sleep(random.uniform(30, 180))
    async def _generate_signal(self):
        """Generate a mock signal"""
        try:
            current_regime = self.orchestrator.current_regime
            regime_multiplier = self.regime_preferences.get(current_regime, 0.5)
            base_strength = self.base_performance * regime_multiplier
            noise = random.gauss(0, 0.2)
            signal_strength = max(0.1, min(1.0, base_strength + noise))
            expected_return = random.gauss(0.01, 0.02) * signal_strength
            confidence = signal_strength * random.uniform(0.8, 1.2)
            confidence = max(0.1, min(1.0, confidence))
            signal = StrategySignal(
                strategy_id=self.strategy_id,
                strategy_type=self.strategy_type,
                symbol=random.choice(['BTCUSDT', 'ETHUSDT', 'ADAUSDT']),
                signal_strength=signal_strength,
                expected_return=expected_return,
                confidence=confidence,
                holding_period=random.randint(15, 240),
                metadata={
                    'source': 'mock_strategy',
                    'regime_multiplier': regime_multiplier,
                    'base_performance': self.base_performance
                },
                timestamp=time.time()
            )
            await self.orchestrator.process_strategy_signal(signal)
        except Exception as e:
            print(f"❌ Signal generation error: {e}")
    def stop(self):
        """Stop generating signals"""
        self.running = False
class StrategyOrchestrationTester:
    """Test the strategy orchestration engine"""
    def __init__(self):
        self.orchestrator = StrategyOrchestrationEngine(
            initial_capital=100000.0,
            max_strategies=8,
            rebalance_frequency=120
        )
        self.orchestrator.add_signal_handler(self.handle_execution_signal)
        self.orchestrator.add_allocation_handler(self.handle_allocation_update)
        self.orchestrator.add_insight_handler(self.handle_meta_insight)
        self.execution_signals = []
        self.allocation_updates = []
        self.meta_insights = []
        self.strategies = []
    async def handle_execution_signal(self, signal):
        """Handle execution signals from orchestrator"""
        self.execution_signals.append(signal)
        print(f"🎯 Execution Signal: {signal['strategy_id']} - {signal['symbol']} "
              f"(Strength: {signal['signal_strength']:.2f}, Capital: ${signal['allocated_capital']:,.0f})")
    async def handle_allocation_update(self, allocations):
        """Handle capital allocation updates"""
        self.allocation_updates.append(allocations)
        print(f"⚖️ Capital Allocation Update:")
        for strategy_id, allocation in allocations.items():
            print(f"   {strategy_id}: {allocation:.1%}")
    async def handle_meta_insight(self, insight):
        """Handle meta-learning insights"""
        self.meta_insights.append(insight)
        print(f"🔬 Meta-Learning Insight: {insight.insight_type}")
        print(f"   {insight.description} (Confidence: {insight.confidence:.1%})")
    async def setup_mock_strategies(self):
        """Set up mock strategies for testing"""
        strategy_configs = [
            ("momentum_1", StrategyType.PRICE_MOMENTUM),
            ("sentiment_1", StrategyType.SENTIMENT_ALPHA),
            ("fusion_1", StrategyType.FUSION_SIGNALS),
            ("ml_alpha_1", StrategyType.ML_ALPHA),
            ("mean_reversion_1", StrategyType.MEAN_REVERSION),
            ("momentum_2", StrategyType.PRICE_MOMENTUM),
        ]
        print("🤖 Setting up mock strategies...")
        for strategy_id, strategy_type in strategy_configs:
            strategy = MockStrategy(strategy_id, strategy_type, self.orchestrator)
            self.strategies.append(strategy)
            print(f"   ✅ {strategy_id} ({strategy_type.value})")
        print(f"📊 Created {len(self.strategies)} mock strategies")
    async def run_orchestration_test(self):
        """Run comprehensive orchestration test"""
        print("\n🧠 Starting Strategy Orchestration Test")
        print("=" * 50)
        strategy_tasks = []
        for strategy in self.strategies:
            task = asyncio.create_task(strategy.start())
            strategy_tasks.append(task)
        test_duration = 300
        start_time = time.time()
        print(f"🚀 Running orchestration for {test_duration} seconds...")
        print("📊 Watch for signals, allocations, and insights...\n")
        while time.time() - start_time < test_duration:
            await asyncio.sleep(30)
            elapsed = time.time() - start_time
            print(f"⏱️  Elapsed: {elapsed:.0f}s - Signals: {len(self.execution_signals)}, "
                  f"Allocations: {len(self.allocation_updates)}, Insights: {len(self.meta_insights)}")
        for strategy in self.strategies:
            strategy.stop()
        for task in strategy_tasks:
            task.cancel()
        await asyncio.sleep(2)
    def analyze_orchestration_results(self):
        """Analyze orchestration test results"""
        print("\n📈 Orchestration Test Results")
        print("=" * 50)
        summary = self.orchestrator.get_orchestration_summary()
        print(f"🧠 Orchestration Summary:")
        print(f"   Current Regime: {summary['current_regime']}")
        print(f"   Total Strategies: {summary['total_strategies']}")
        print(f"   Signals Processed Today: {summary['signals_processed_today']}")
        print(f"   Meta-Learning Insights: {summary['meta_learning_insights_count']}")
        print(f"\n📊 Strategy Performance:")
        for strategy_id, perf in summary['strategy_performance'].items():
            print(f"   {strategy_id}:")
            print(f"     Type: {perf['type']}")
            print(f"     Allocation: {perf['allocation']:.1%}")
            print(f"     Recent Performance: {perf['recent_performance']:.3f}")
            print(f"     Total Signals: {perf['total_signals']}")
            print(f"     Win Rate: {perf['win_rate']:.1%}")
        print(f"\n🎯 Execution Signals Generated: {len(self.execution_signals)}")
        if self.execution_signals:
            print("   Recent signals:")
            for signal in self.execution_signals[-3:]:
                print(f"     {signal['strategy_id']} - {signal['symbol']} "
                      f"(${signal['allocated_capital']:,.0f})")
        print(f"\n⚖️ Capital Allocation Updates: {len(self.allocation_updates)}")
        if self.allocation_updates:
            latest_allocation = self.allocation_updates[-1]
            print("   Latest allocation:")
            for strategy_id, allocation in latest_allocation.items():
                print(f"     {strategy_id}: {allocation:.1%}")
        print(f"\n🔬 Meta-Learning Insights: {len(self.meta_insights)}")
        for insight in self.meta_insights[-3:]:
            print(f"   {insight.insight_type}: {insight.description}")
        print(f"\n📋 Orchestration Metrics:")
        metrics = summary['orchestration_metrics']
        for key, value in metrics.items():
            print(f"   {key}: {value}")
async def main():
    print("🧠 Strategy Orchestration Engine Test Suite")
    print("=" * 60)
    tester = StrategyOrchestrationTester()
    try:
        await tester.setup_mock_strategies()
        await tester.run_orchestration_test()
        tester.analyze_orchestration_results()
        tester.orchestrator.save_orchestration_state()
        print("\n✅ Strategy Orchestration Test Complete!")
    except KeyboardInterrupt:
        print("\n🛑 Test stopped by user")
        for strategy in tester.strategies:
            strategy.stop()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    asyncio.run(main())