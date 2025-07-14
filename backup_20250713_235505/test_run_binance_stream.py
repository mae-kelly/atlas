import asyncio
from data_streams.price_feed.binance_ws_price_stream import BinancePriceStream
async def handle_tick(tick):
    symbol = tick.get("s")
    price = tick.get("p")
    qty = tick.get("q")
    ts = tick.get("T")
    print(f"[{symbol}] Price: {price} Qty: {qty} Timestamp: {ts}")
async def main():
    stream = BinancePriceStream(
        pairs=["BTCUSDT", "ETHUSDT"], 
        on_message=handle_tick
    )
    await stream.connect()
if __name__ == "__main__":
    asyncio.run(main())