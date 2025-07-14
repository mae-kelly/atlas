import asyncio
import time
import random
import numpy as np
from portfolio.portfolio_management_engine import PortfolioManagementEngine
class PortfolioManagementTester:
    """Test the portfolio management engine"""
    def __init__(self):
        self.portfolio_engine = PortfolioManagementEngine(
            initial_capital=100000.0,
            benchmark_symbol="BTC",
            performance_window=252
        )
        self.portfolio_engine.add_portfolio_update_handler(self.handle_portfolio_update)
        self.portfolio_engine.add_performance_alert_handler(self.handle_performance_alert)
        self.received_updates = []
        self.received_alerts = []
    async def handle_portfolio_update(self, snapshot):
        """Handle portfolio updates"""
        self.received_updates.append(snapshot)
        print(f"📊 Portfolio Update: Value=${snapshot.total_value:,.2f}, "
              f"Return={snapshot.cumulative_return:.2%}, Positions={snapshot.number_of_positions}")
    async def handle_performance_alert(self, alert):
        """Handle performance alerts"""
        self.received_alerts.append(alert)
        print(f"🚨 Performance Alert: {alert['type']} - {alert['message']}")
    async def simulate_trading_activity(self):
        """Simulate realistic trading activity"""
        print("\n📈 Simulating Trading Activity")
        print("=" * 40)
        trades = [
            ("BTCUSDT", 1.0, 45000, "buy", "alpha_strategy_1"),
            ("ETHUSDT", 5.0, 3000, "buy", "alpha_strategy_2"),
            ("BTCUSDT", 0.5, 46000, "buy", "alpha_strategy_1"),
            ("ADAUSDT", 1000, 1.2, "buy", "alpha_strategy_3"),
            ("BTCUSDT", -0.3, 47000, "sell", "profit_taking"),
            ("ETHUSDT", -2.0, 3100, "sell", "rebalance"),
            ("DOTUSDT", 100, 25, "buy", "alpha_strategy_4"),
            ("BTCUSDT", -0.5, 48000, "sell", "alpha_strategy_1"),
        ]
        for i, (symbol, quantity, price, trade_type, strategy) in enumerate(trades):
            print(f"  Trade {i+1}: {trade_type} {abs(quantity)} {symbol} @ ${price}")
            await self.portfolio_engine.update_position(
                symbol=symbol,
                quantity=quantity,
                price=price,
                trade_type=trade_type,
                strategy_id=strategy
            )
            await asyncio.sleep(0.5)
        print(f"✅ Executed {len(trades)} trades")
    async def simulate_market_movements(self):
        """Simulate market price movements"""
        print("\n📊 Simulating Market Movements")
        print("=" * 40)
        prices = {
            "BTCUSDT": 45000,
            "ETHUSDT": 3000,
            "ADAUSDT": 1.2,
            "DOTUSDT": 25
        }
        for day in range(50):
            updated_prices = {}
            for symbol, base_price in prices.items():
                daily_return = random.gauss(0, 0.03)
                if day < 25:
                    daily_return += 0.001
                else:
                    daily_return -= 0.0005
                new_price = base_price * (1 + daily_return)
                prices[symbol] = new_price
                updated_prices[symbol] = new_price
            await self.portfolio_engine.update_market_prices(updated_prices)
            if day % 10 == 0:
                print(f"  Day {day}: BTC=${prices['BTCUSDT']:,.2f}, ETH=${prices['ETHUSDT']:,.2f}")
            await asyncio.sleep(0.1)
        print("✅ Completed 50 days of market simulation")
    async def test_risk_monitoring(self):
        """Test risk monitoring features"""
        print("\n🛡️ Testing Risk Monitoring")
        print("=" * 40)
        crash_prices = {}
        current_positions = list(self.portfolio_engine.positions.keys())
        for symbol in current_positions:
            current_price = self.portfolio_engine.positions[symbol].current_price
            crash_prices[symbol] = current_price * 0.7
        print("  Simulating market crash (-30%)...")
        await self.portfolio_engine.update_market_prices(crash_prices)
        await asyncio.sleep(1)
        recovery_prices = {}
        for symbol in current_positions:
            current_price = self.portfolio_engine.positions[symbol].current_price
            recovery_prices[symbol] = current_price * 1.15
        print("  Simulating partial recovery (+15%)...")
        await self.portfolio_engine.update_market_prices(recovery_prices)
        await asyncio.sleep(1)
    async def test_rebalancing(self):
        """Test portfolio rebalancing"""
        print("\n⚖️ Testing Portfolio Rebalancing")
        print("=" * 40)
        target_weights = {
            "BTCUSDT": 0.4,   # 40% BTC
            "ETHUSDT": 0.3,   # 30% ETH
            "ADAUSDT": 0.2,   # 20% ADA
            "DOTUSDT": 0.1    # 10% DOT
        }
        print("  Target allocation:")
        for symbol, weight in target_weights.items():
            print(f"    {symbol}: {weight:.1%}")
        rebalance_orders = await self.portfolio_engine.rebalance_portfolio(target_weights)
        print(f"\n  Generated {len(rebalance_orders)} rebalancing orders:")
        for order in rebalance_orders:
            direction = "BUY" if order['quantity'] > 0 else "SELL"
            print(f"    {direction} {abs(order['quantity']):.3f} {order['symbol']} "
                  f"(target: {order['target_weight']:.1%})")
    def analyze_performance(self):
        """Analyze final performance"""
        print("\n📈 Performance Analysis")
        print("=" * 40)
        summary = self.portfolio_engine.get_portfolio_summary()
        print(f"Initial Capital: ${self.portfolio_engine.initial_capital:,.2f}")
        print(f"Final Value: ${summary['total_value']:,.2f}")
        print(f"Total Return: {summary['cumulative_return']:.2%}")
        print(f"Cash Remaining: ${summary['cash']:,.2f}")
        print(f"Unrealized P&L: ${summary['total_unrealized_pnl']:,.2f}")
        print(f"Realized P&L: ${summary['total_realized_pnl']:,.2f}")
        print("\n📊 Current Positions:")
        for symbol, pos_data in summary['positions'].items():
            print(f"  {symbol}: {pos_data['quantity']:.3f} @ ${pos_data['current_price']:,.2f} "
                  f"(Weight: {pos_data['weight']:.1%}, P&L: ${pos_data['unrealized_pnl']:,.2f})")
        trade_summary = summary['trade_summary']
        print(f"\n📝 Trading Summary:")
        print(f"  Total Trades: {trade_summary['total_trades']}")
        print(f"  Win Rate: {trade_summary['win_rate']:.1%}")
        print(f"  Avg Trade Duration: {trade_summary['average_trade_duration']:.1f} hours")
        if summary['performance_metrics']:
            metrics = summary['performance_metrics']
            print(f"\n🎯 Performance Metrics:")
            print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A'):.2f}")
            print(f"  Max Drawdown: {metrics.get('max_drawdown', 'N/A'):.2%}")
            print(f"  Volatility: {metrics.get('volatility', 'N/A'):.2%}")
            print(f"  Sortino Ratio: {metrics.get('sortino_ratio', 'N/A'):.2f}")
        print(f"\n🚨 Risk Alerts: {len(self.received_alerts)} total")
        for alert in self.received_alerts[-3:]:
            print(f"  {alert['type']}: {alert['message']}")
        print("\n📄 Generating Performance Report...")
        report = self.portfolio_engine.get_performance_report(period_days=30)
        if 'trade_analysis' in report:
            trade_analysis = report['trade_analysis']
            print(f"  Recent Performance (30 days):")
            print(f"    Trades: {trade_analysis.get('total_trades', 0)}")
            print(f"    Win Rate: {trade_analysis.get('win_rate', 0):.1%}")
            print(f"    Total P&L: ${trade_analysis.get('total_pnl', 0):,.2f}")
async def main():
    print("📊 Portfolio Management Engine Test Suite")
    print("=" * 50)
    tester = PortfolioManagementTester()
    try:
        await tester.simulate_trading_activity()
        await tester.simulate_market_movements()
        await tester.test_risk_monitoring()
        await tester.test_rebalancing()
        tester.analyze_performance()
        tester.portfolio_engine.save_performance_data()
        print("\n✅ Portfolio Management Test Complete!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    asyncio.run(main())