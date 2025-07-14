import asyncio
import time
import json
from typing import Dict, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from loguru import logger
from collections import deque, defaultdict
import numpy as np
class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"
class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"
class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
class ExecutionUrgency(Enum):
    LOW = "low"         # Minimize market impact
    NORMAL = "normal"   # Balance speed vs impact
    HIGH = "high"       # Prioritize speed
    EMERGENCY = "emergency"  # Market orders, immediate execution
@dataclass
class OrderRequest:
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Cancelled
    urgency: ExecutionUrgency = ExecutionUrgency.NORMAL
    max_slippage: float = 0.005
    execution_strategy: str = "smart"
    parent_strategy_id: Optional[str] = None
    metadata: Dict = None
@dataclass
class OrderFill:
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    fee: float
    fee_currency: str
    timestamp: float
    exchange_fill_id: str
@dataclass
class Order:
    order_id: str
    request: OrderRequest
    status: OrderStatus
    exchange_order_id: Optional[str] = None
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    average_fill_price: float = 0.0
    total_fees: float = 0.0
    fills: List[OrderFill] = None
    creation_time: float = 0.0
    last_update_time: float = 0.0
    error_message: Optional[str] = None
    def __post_init__(self):
        if self.fills is None:
            self.fills = []
        if self.creation_time == 0.0:
            self.creation_time = time.time()
        if self.remaining_quantity == 0.0:
            self.remaining_quantity = self.request.quantity
@dataclass
class ExecutionReport:
    order_id: str
    symbol: str
    requested_quantity: float
    executed_quantity: float
    average_price: float
    total_slippage: float
    total_fees: float
    execution_time_seconds: float
    market_impact: float
    execution_quality_score: float
    strategy_used: str
    fills: List[OrderFill]
    timestamp: float
class OrderExecutionEngine:
    """
    Advanced order execution engine with smart routing and execution algorithms
    """
    def __init__(self, 
                 exchanges: List = None,
                 default_slippage_tolerance: float = 0.005,
                 max_order_size_usd: float = 50000):
        self.exchanges = exchanges or []
        self.default_slippage_tolerance = default_slippage_tolerance
        self.max_order_size_usd = max_order_size_usd
        self.orders: Dict[str, Order] = {}
        self.execution_history: deque = deque(maxlen=1000)
        self.execution_strategies = {}
        self._initialize_execution_strategies()
        self.market_data = {}
        self.order_book_data = defaultdict(dict)
        self.execution_metrics = {
            'total_orders': 0,
            'successful_executions': 0,
            'average_slippage': 0.0,
            'average_execution_time': 0.0,
            'total_fees_paid': 0.0
        }
        self.order_update_handlers = []
        self.execution_report_handlers = []
        self.rate_limiters = {}
        logger.info("⚡ Order Execution Engine initialized")
    def add_exchange(self, exchange):
        """Add exchange connector"""
        self.exchanges.append(exchange)
        logger.info(f"📈 Added exchange: {exchange.name}")
    def add_order_update_handler(self, handler: Callable):
        """Add callback for order updates"""
        self.order_update_handlers.append(handler)
    def add_execution_report_handler(self, handler: Callable):
        """Add callback for execution reports"""
        self.execution_report_handlers.append(handler)
    async def execute_order(self, order_request: OrderRequest) -> str:
        """
        Execute an order using optimal execution strategy
        """
        try:
            order_id = str(uuid.uuid4())
            order = Order(
                order_id=order_id,
                request=order_request,
                status=OrderStatus.PENDING,
                remaining_quantity=order_request.quantity
            )
            self.orders[order_id] = order
            self.execution_metrics['total_orders'] += 1
            logger.info(f"🎯 New Order: {order_id} - {order_request.symbol} "
                       f"{order_request.side.value} {order_request.quantity}")
            execution_strategy = self._select_execution_strategy(order_request)
            await self._execute_with_strategy(order, execution_strategy)
            return order_id
        except Exception as e:
            logger.error(f"❌ Order execution error: {e}")
            if 'order' in locals():
                order.status = OrderStatus.REJECTED
                order.error_message = str(e)
                await self._notify_order_update(order)
            raise
    def _select_execution_strategy(self, order_request: OrderRequest) -> str:
        """
        Select optimal execution strategy based on order characteristics
        """
        if order_request.execution_strategy != "smart":
            return order_request.execution_strategy
        if order_request.urgency == ExecutionUrgency.EMERGENCY:
            return "market_rush"
        elif order_request.urgency == ExecutionUrgency.HIGH:
            return "aggressive_limit"
        elif order_request.urgency == ExecutionUrgency.LOW:
            return "iceberg_passive"
        else:
            estimated_impact = self._estimate_market_impact(order_request)
            if estimated_impact > 0.01:
                return "iceberg_twap"
            elif estimated_impact > 0.005:
                return "adaptive_limit"
            else:
                return "smart_limit"
    async def _execute_with_strategy(self, order: Order, strategy_name: str):
        """
        Execute order using specified strategy
        """
        try:
            order.status = OrderStatus.SUBMITTED
            await self._notify_order_update(order)
            strategy = self.execution_strategies.get(strategy_name)
            if not strategy:
                raise ValueError(f"Unknown execution strategy: {strategy_name}")
            start_time = time.time()
            await strategy(order)
            execution_time = time.time() - start_time
            await self._generate_execution_report(order, strategy_name, execution_time)
        except Exception as e:
            logger.error(f"❌ Strategy execution error: {e}")
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            await self._notify_order_update(order)
    def _initialize_execution_strategies(self):
        """Initialize execution strategy functions"""
        async def market_rush_strategy(order: Order):
            """Emergency market order execution"""
            await self._execute_market_order(order)
        async def aggressive_limit_strategy(order: Order):
            """Aggressive limit orders walking the book"""
            await self._execute_aggressive_limit(order)
        async def smart_limit_strategy(order: Order):
            """Smart limit order with adaptive pricing"""
            await self._execute_smart_limit(order)
        async def iceberg_passive_strategy(order: Order):
            """Large order broken into small passive pieces"""
            await self._execute_iceberg_passive(order)
        async def iceberg_twap_strategy(order: Order):
            """TWAP execution with iceberg orders"""
            await self._execute_iceberg_twap(order)
        async def adaptive_limit_strategy(order: Order):
            """Adaptive limit orders with market monitoring"""
            await self._execute_adaptive_limit(order)
        self.execution_strategies = {
            'market_rush': market_rush_strategy,
            'aggressive_limit': aggressive_limit_strategy,
            'smart_limit': smart_limit_strategy,
            'iceberg_passive': iceberg_passive_strategy,
            'iceberg_twap': iceberg_twap_strategy,
            'adaptive_limit': adaptive_limit_strategy
        }
    async def _execute_market_order(self, order: Order):
        """Execute market order for immediate fill"""
        try:
            exchange = await self._select_best_exchange(order.request.symbol)
            estimated_slippage = await self._estimate_slippage(order.request, exchange)
            if estimated_slippage > order.request.max_slippage:
                logger.warning(f"⚠️ High slippage estimate: {estimated_slippage:.2%}")
            exchange_order = await exchange.create_market_order(
                order.request.symbol,
                order.request.side.value,
                order.request.quantity
            )
            order.exchange_order_id = exchange_order['id']
            order.status = OrderStatus.SUBMITTED
            await self._monitor_order_fills(order, exchange)
        except Exception as e:
            logger.error(f"❌ Market order execution failed: {e}")
            raise
    async def _execute_smart_limit(self, order: Order):
        """Execute smart limit order with optimal pricing"""
        try:
            exchange = await self._select_best_exchange(order.request.symbol)
            ticker = await exchange.fetch_ticker(order.request.symbol)
            order_book = await exchange.fetch_order_book(order.request.symbol)
            limit_price = self._calculate_smart_limit_price(
                order.request, ticker, order_book
            )
            exchange_order = await exchange.create_limit_order(
                order.request.symbol,
                order.request.side.value,
                order.request.quantity,
                limit_price
            )
            order.exchange_order_id = exchange_order['id']
            order.status = OrderStatus.SUBMITTED
            await self._monitor_and_adjust_limit_order(order, exchange)
        except Exception as e:
            logger.error(f"❌ Smart limit execution failed: {e}")
            raise
    async def _execute_iceberg_twap(self, order: Order):
        """Execute large order using TWAP with iceberg strategy"""
        try:
            total_quantity = order.request.quantity
            execution_window = 300
            slice_count = 10
            slice_size = total_quantity / slice_count
            logger.info(f"📊 TWAP Execution: {slice_count} slices of {slice_size}")
            for i in range(slice_count):
                if order.status in [OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                    break
                slice_order = OrderRequest(
                    symbol=order.request.symbol,
                    side=order.request.side,
                    quantity=slice_size,
                    order_type=OrderType.LIMIT,
                    urgency=ExecutionUrgency.LOW
                )
                await self._execute_smart_limit_slice(order, slice_order, i + 1)
                if i < slice_count - 1:
                    await asyncio.sleep(execution_window / slice_count)
        except Exception as e:
            logger.error(f"❌ TWAP execution failed: {e}")
            raise
    async def _execute_smart_limit_slice(self, parent_order: Order, slice_order: OrderRequest, slice_num: int):
        """Execute a single slice of an iceberg order"""
        try:
            exchange = await self._select_best_exchange(slice_order.symbol)
            ticker = await exchange.fetch_ticker(slice_order.symbol)
            order_book = await exchange.fetch_order_book(slice_order.symbol)
            limit_price = self._calculate_smart_limit_price(slice_order, ticker, order_book)
            exchange_order = await exchange.create_limit_order(
                slice_order.symbol,
                slice_order.side.value,
                slice_order.quantity,
                limit_price
            )
            logger.info(f"🧊 Iceberg slice {slice_num}: {slice_order.quantity} @ {limit_price}")
            await self._monitor_slice_execution(parent_order, exchange_order, exchange)
        except Exception as e:
            logger.error(f"❌ Slice execution failed: {e}")
            raise
    def _calculate_smart_limit_price(self, order_request: OrderRequest, ticker: Dict, order_book: Dict) -> float:
        """
        Calculate optimal limit price based on market conditions
        """
        try:
            bid = ticker['bid']
            ask = ticker['ask']
            spread = ask - bid
            mid = (bid + ask) / 2
            if order_request.side == OrderSide.BUY:
                if order_request.urgency == ExecutionUrgency.HIGH:
                    price = bid + spread * 0.7
                elif order_request.urgency == ExecutionUrgency.LOW:
                    price = bid + spread * 0.2
                else:
                    price = bid + spread * 0.4
            else:
                if order_request.urgency == ExecutionUrgency.HIGH:
                    price = ask - spread * 0.7
                elif order_request.urgency == ExecutionUrgency.LOW:
                    price = ask - spread * 0.2
                else:
                    price = ask - spread * 0.4
            impact_adjustment = self._calculate_market_impact_adjustment(order_request, order_book)
            if order_request.side == OrderSide.BUY:
                price += impact_adjustment
            else:
                price -= impact_adjustment
            max_deviation = mid * 0.01
            price = max(mid - max_deviation, min(mid + max_deviation, price))
            return round(price, 8)
        except Exception as e:
            logger.error(f"❌ Price calculation error: {e}")
            return (ticker['bid'] + ticker['ask']) / 2
    def _calculate_market_impact_adjustment(self, order_request: OrderRequest, order_book: Dict) -> float:
        """
        Calculate price adjustment for market impact
        """
        try:
            if order_request.side == OrderSide.BUY:
                asks = order_book.get('asks', [])
                total_ask_volume = sum(ask[1] for ask in asks[:10])
            else:
                bids = order_book.get('bids', [])
                total_bid_volume = sum(bid[1] for bid in bids[:10])
                total_ask_volume = total_bid_volume
            if total_ask_volume > 0:
                impact_ratio = order_request.quantity / total_ask_volume
                impact_adjustment = impact_ratio * 0.001
                return min(impact_adjustment, 0.005)
            return 0.0
        except Exception as e:
            logger.error(f"❌ Market impact calculation error: {e}")
            return 0.0
    async def _monitor_order_fills(self, order: Order, exchange):
        """Monitor order for fills and update status"""
        try:
            max_wait_time = 300
            start_time = time.time()
            while (time.time() - start_time) < max_wait_time:
                exchange_order = await exchange.fetch_order(order.exchange_order_id, order.request.symbol)
                await self._update_order_from_exchange(order, exchange_order)
                if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                    break
                await asyncio.sleep(1)
            if order.status not in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                logger.warning(f"⏰ Order timeout: {order.order_id}")
                await self._handle_order_timeout(order, exchange)
        except Exception as e:
            logger.error(f"❌ Order monitoring error: {e}")
    async def _update_order_from_exchange(self, order: Order, exchange_order: Dict):
        """Update order object with exchange data"""
        try:
            exchange_status = exchange_order.get('status', '').lower()
            if exchange_status == 'closed':
                order.status = OrderStatus.FILLED
            elif exchange_status == 'canceled':
                order.status = OrderStatus.CANCELLED
            elif exchange_status == 'rejected':
                order.status = OrderStatus.REJECTED
            elif exchange_order.get('filled', 0) > 0:
                order.status = OrderStatus.PARTIALLY_FILLED
            filled_quantity = exchange_order.get('filled', 0)
            if filled_quantity > order.filled_quantity:
                new_fill_qty = filled_quantity - order.filled_quantity
                new_fill_price = exchange_order.get('average', exchange_order.get('price', 0))
                fill = OrderFill(
                    fill_id=str(uuid.uuid4()),
                    order_id=order.order_id,
                    symbol=order.request.symbol,
                    side=order.request.side,
                    quantity=new_fill_qty,
                    price=new_fill_price,
                    fee=exchange_order.get('fee', {}).get('cost', 0),
                    fee_currency=exchange_order.get('fee', {}).get('currency', 'USDT'),
                    timestamp=time.time(),
                    exchange_fill_id=exchange_order.get('id', '')
                )
                order.fills.append(fill)
                order.filled_quantity = filled_quantity
                order.remaining_quantity = order.request.quantity - filled_quantity
                if order.filled_quantity > 0:
                    total_value = sum(fill.quantity * fill.price for fill in order.fills)
                    order.average_fill_price = total_value / order.filled_quantity
                order.total_fees += fill.fee
                logger.info(f"💰 Fill: {order.order_id} - {new_fill_qty} @ {new_fill_price}")
            order.last_update_time = time.time()
            await self._notify_order_update(order)
        except Exception as e:
            logger.error(f"❌ Order update error: {e}")
    async def _notify_order_update(self, order: Order):
        """Notify all order update handlers"""
        for handler in self.order_update_handlers:
            try:
                await asyncio.create_task(handler(order))
            except Exception as e:
                logger.error(f"❌ Order update handler error: {e}")
    async def _generate_execution_report(self, order: Order, strategy_name: str, execution_time: float):
        """Generate comprehensive execution report"""
        try:
            if order.filled_quantity > 0 and hasattr(order.request, 'expected_price'):
                expected_price = order.request.expected_price
                actual_price = order.average_fill_price
                slippage = (actual_price - expected_price) / expected_price
            else:
                slippage = 0.0
            quality_score = self._calculate_execution_quality_score(order, slippage, execution_time)
            report = ExecutionReport(
                order_id=order.order_id,
                symbol=order.request.symbol,
                requested_quantity=order.request.quantity,
                executed_quantity=order.filled_quantity,
                average_price=order.average_fill_price,
                total_slippage=slippage,
                total_fees=order.total_fees,
                execution_time_seconds=execution_time,
                market_impact=self._estimate_market_impact(order.request),
                execution_quality_score=quality_score,
                strategy_used=strategy_name,
                fills=order.fills.copy(),
                timestamp=time.time()
            )
            self.execution_history.append(report)
            self._update_execution_metrics(report)
            await self._notify_execution_report(report)
            logger.info(f"📈 Execution Report: {order.order_id} - Quality: {quality_score:.2f}")
        except Exception as e:
            logger.error(f"❌ Execution report generation error: {e}")
    def _calculate_execution_quality_score(self, order: Order, slippage: float, execution_time: float) -> float:
        """
        Calculate execution quality score (0-1, higher is better)
        """
        try:
            score = 1.0
            slippage_penalty = min(abs(slippage) * 10, 0.5)
            score -= slippage_penalty
            if order.request.urgency in [ExecutionUrgency.HIGH, ExecutionUrgency.EMERGENCY]:
                time_penalty = min(execution_time / 60, 0.3)
                score -= time_penalty
            fill_rate = order.filled_quantity / order.request.quantity
            score *= fill_rate
            if order.total_fees > 0 and order.filled_quantity > 0:
                fee_rate = order.total_fees / (order.filled_quantity * order.average_fill_price)
                fee_penalty = min(fee_rate * 20, 0.2)
                score -= fee_penalty
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.error(f"❌ Quality score calculation error: {e}")
            return 0.5
    async def _select_best_exchange(self, symbol: str):
        """Select best exchange for trading symbol"""
        if self.exchanges:
            return self.exchanges[0]
        else:
            raise ValueError("No exchanges available")
    async def _estimate_slippage(self, order_request: OrderRequest, exchange) -> float:
        """Estimate execution slippage"""
        try:
            order_book = await exchange.fetch_order_book(order_request.symbol)
            spread = order_book['asks'][0][0] - order_book['bids'][0][0]
            mid_price = (order_book['asks'][0][0] + order_book['bids'][0][0]) / 2
            spread_pct = spread / mid_price
            impact = self._estimate_market_impact(order_request)
            estimated_slippage = spread_pct * 0.5 + impact
            return estimated_slippage
        except Exception:
            return 0.001
    def _estimate_market_impact(self, order_request: OrderRequest) -> float:
        """Estimate market impact of order"""
        estimated_price = self.market_data.get(order_request.symbol, {}).get('price', 50000)
        order_value_usd = order_request.quantity * estimated_price
        if order_value_usd < 1000:
            return 0.0001
        elif order_value_usd < 10000:
            return 0.0005
        elif order_value_usd < 50000:
            return 0.002
        else:
            return 0.005
    async def _handle_order_timeout(self, order: Order, exchange):
        """Handle order timeout"""
        try:
            await exchange.cancel_order(order.exchange_order_id, order.request.symbol)
            order.status = OrderStatus.CANCELLED
            await self._notify_order_update(order)
        except Exception as e:
            logger.error(f"❌ Order timeout handling error: {e}")
    async def _monitor_and_adjust_limit_order(self, order: Order, exchange):
        """Monitor limit order and adjust if needed"""
        await self._monitor_order_fills(order, exchange)
    async def _monitor_slice_execution(self, parent_order: Order, exchange_order: Dict, exchange):
        """Monitor execution of iceberg slice"""
        slice_order = Order(
            order_id=f"{parent_order.order_id}_slice",
            request=parent_order.request,
            status=OrderStatus.SUBMITTED,
            exchange_order_id=exchange_order['id']
        )
        await self._monitor_order_fills(slice_order, exchange)
        parent_order.filled_quantity += slice_order.filled_quantity
        parent_order.fills.extend(slice_order.fills)
        if parent_order.filled_quantity >= parent_order.request.quantity:
            parent_order.status = OrderStatus.FILLED
    def _update_execution_metrics(self, report: ExecutionReport):
        """Update execution performance metrics"""
        self.execution_metrics['successful_executions'] += 1
        total = self.execution_metrics['successful_executions']
        self.execution_metrics['average_slippage'] = (
            (self.execution_metrics['average_slippage'] * (total - 1) + abs(report.total_slippage)) / total
        )
        self.execution_metrics['average_execution_time'] = (
            (self.execution_metrics['average_execution_time'] * (total - 1) + report.execution_time_seconds) / total
        )
        self.execution_metrics['total_fees_paid'] += report.total_fees
    async def _notify_execution_report(self, report: ExecutionReport):
        """Notify execution report handlers"""
        for handler in self.execution_report_handlers:
            try:
                await asyncio.create_task(handler(report))
            except Exception as e:
                logger.error(f"❌ Execution report handler error: {e}")
    def get_execution_summary(self) -> Dict:
        """Get execution performance summary"""
        success_rate = (
            self.execution_metrics['successful_executions'] / self.execution_metrics['total_orders']
            if self.execution_metrics['total_orders'] > 0 else 0
        )
        return {
            'total_orders': self.execution_metrics['total_orders'],
            'successful_executions': self.execution_metrics['successful_executions'],
            'success_rate': success_rate,
            'average_slippage': self.execution_metrics['average_slippage'],
            'average_execution_time': self.execution_metrics['average_execution_time'],
            'total_fees_paid': self.execution_metrics['total_fees_paid'],
            'active_orders': len([o for o in self.orders.values() if o.status not in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]]),
            'recent_reports': list(self.execution_history)[-5:]  # Last 5 execution reports
        }
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order"""
        try:
            if order_id not in self.orders:
                return False
            order = self.orders[order_id]
            if order.status not in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
                return False
            exchange = await self._select_best_exchange(order.request.symbol)
            await exchange.cancel_order(order.exchange_order_id, order.request.symbol)
            order.status = OrderStatus.CANCELLED
            await self._notify_order_update(order)
            logger.info(f"❌ Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Order cancellation error: {e}")
            return False