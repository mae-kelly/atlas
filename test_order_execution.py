import asyncio
import time
from execution.order_execution_engine import OrderExecutionEngine, OrderRequest, OrderSide, OrderType, ExecutionUrgency
from execution.exchanges.exchange_connector import MockExchange

class OrderExecutionTester:
    """Test the order execution engine"""
    
    def __init__(self):
        # Initialize mock exchange
        self.mock_exchange = MockExchange()
        
        # Initialize execution engine
        self.execution_engine = OrderExecutionEngine(
            exchanges=[self.mock_exchange],
            default_slippage_tolerance=0.005,
            max_order_size_usd=50000
        )
        
        # Add handlers
        self.execution_engine.add_order_update_handler(self.handle_order_update)
        self.execution_engine.add_execution_report_handler(self.handle_execution_report)
        
        self.received_updates = []
        self.received_reports = []
    
    async def handle_order_update(self, order):
        """Handle order status updates"""
        self.received_updates.append(order)
        print(f"📨 Order Update: {order.order_id} - {order.status.value}")
        if order.filled_quantity > 0:
            print(f"   Filled: {order.filled_quantity} @ avg {order.average_fill_price}")
    
    async def handle_execution_report(self, report):
        """Handle execution reports"""
        self.received_reports.append(report)
        print(f"📊 Execution Report: {report.order_id}")
        print(f"   Quality Score: {report.execution_quality_score:.2f}")
        print(f"   Slippage: {report.total_slippage:.4f}")
        print(f"   Execution Time: {report.execution_time_seconds:.1f}s")
    
    async def test_market_orders(self):
        """Test market order execution"""
        print("\n🎯 Testing Market Orders")
        print("=" * 30)
        
        # Test market buy order
        buy_request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=0.1,
            order_type=OrderType.MARKET,
            urgency=ExecutionUrgency.HIGH
        )
        
        buy_order_id = await self.execution_engine.execute_order(buy_request)
        print(f"✅ Market buy order submitted: {buy_order_id}")
        
        # Wait for execution
        await asyncio.sleep(3)
        
        # Test market sell order
        sell_request = OrderRequest(
            symbol="ETHUSDT",
            side=OrderSide.SELL,
            quantity=1.0,
            order_type=OrderType.MARKET,
            urgency=ExecutionUrgency.EMERGENCY
        )
        
        sell_order_id = await self.execution_engine.execute_order(sell_request)
        print(f"✅ Market sell order submitted: {sell_order_id}")
        
        await asyncio.sleep(3)
    
    async def test_limit_orders(self):
        """Test limit order execution"""
        print("\n📋 Testing Limit Orders")
        print("=" * 30)
        
        # Test aggressive limit order
        aggressive_request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=0.05,
            order_type=OrderType.LIMIT,
            urgency=ExecutionUrgency.HIGH,
            execution_strategy="aggressive_limit"
        )
        
        aggressive_order_id = await self.execution_engine.execute_order(aggressive_request)
        print(f"✅ Aggressive limit order submitted: {aggressive_order_id}")
        
        await asyncio.sleep(3)
        
        # Test passive limit order
        passive_request = OrderRequest(
            symbol="ETHUSDT",
            side=OrderSide.SELL,
            quantity=0.5,
            order_type=OrderType.LIMIT,
            urgency=ExecutionUrgency.LOW,
            execution_strategy="smart_limit"
        )
        
        passive_order_id = await self.execution_engine.execute_order(passive_request)
        print(f"✅ Passive limit order submitted: {passive_order_id}")
        
        await asyncio.sleep(3)
    
    async def test_iceberg_orders(self):
        """Test iceberg/TWAP execution"""
        print("\n🧊 Testing Iceberg/TWAP Orders")
        print("=" * 30)
        
        # Test large order with TWAP
        large_request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=1.0,  # Large order
            order_type=OrderType.TWAP,
            urgency=ExecutionUrgency.LOW,
            execution_strategy="iceberg_twap",
            max_slippage=0.01
        )
        
        twap_order_id = await self.execution_engine.execute_order(large_request)
        print(f"✅ TWAP order submitted: {twap_order_id}")
        
        # Wait longer for TWAP execution
        await asyncio.sleep(10)
    
    async def test_order_cancellation(self):
        """Test order cancellation"""
        print("\n❌ Testing Order Cancellation")
        print("=" * 30)
        
        # Submit limit order
        cancel_request = OrderRequest(
            symbol="ETHUSDT",
            side=OrderSide.BUY,
            quantity=0.2,
            order_type=OrderType.LIMIT,
            urgency=ExecutionUrgency.LOW
        )
        
        cancel_order_id = await self.execution_engine.execute_order(cancel_request)
        print(f"✅ Order to cancel submitted: {cancel_order_id}")
        
        await asyncio.sleep(1)
        
        # Cancel the order
        success = await self.execution_engine.cancel_order(cancel_order_id)
        print(f"{'✅' if success else '❌'} Order cancellation: {success}")
        
        await asyncio.sleep(2)
    
    def analyze_execution_performance(self):
        """Analyze execution performance"""
        print("\n📈 Execution Performance Analysis")
        print("=" * 40)
        
        summary = self.execution_engine.get_execution_summary()
        
        print(f"Total Orders: {summary['total_orders']}")
        print(f"Successful Executions: {summary['successful_executions']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Average Slippage: {summary['average_slippage']:.4f}")
        print(f"Average Execution Time: {summary['average_execution_time']:.1f}s")
        print(f"Total Fees Paid: {summary['total_fees_paid']:.6f}")
        print(f"Active Orders: {summary['active_orders']}")
        
        # Analyze order updates
        print(f"\nOrder Updates Received: {len(self.received_updates)}")
        print(f"Execution Reports Generated: {len(self.received_reports)}")
        
        # Quality score analysis
        if self.received_reports:
            quality_scores = [r.execution_quality_score for r in self.received_reports]
            avg_quality = sum(quality_scores) / len(quality_scores)
            print(f"Average Execution Quality: {avg_quality:.2f}")
        
        print("\n📋 Recent Execution Reports:")
        for report in self.received_reports[-3:]:  # Last 3 reports
            print(f"  {report.symbol} - Strategy: {report.strategy_used}, "
                  f"Quality: {report.execution_quality_score:.2f}, "
                  f"Slippage: {report.total_slippage:.4f}")

async def main():
    print("⚡ Order Execution Engine Test Suite")
    print("=" * 50)
    
    tester = OrderExecutionTester()
    
    try:
        # Connect to mock exchange
        await tester.mock_exchange.connect()
        
        # Run tests
        await tester.test_market_orders()
        await tester.test_limit_orders()
        await tester.test_iceberg_orders()
        await tester.test_order_cancellation()
        
        # Analyze performance
        tester.analyze_execution_performance()
        
        # Disconnect
        await tester.mock_exchange.disconnect()
        
        print("\n✅ Order Execution Test Complete!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
