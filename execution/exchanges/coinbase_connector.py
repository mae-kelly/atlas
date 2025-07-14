import asyncio,aiohttp,ccxt.async_support as ccxt,hmac,hashlib,base64,json,time,os,websockets
from typing import Dict,List,Optional,Tuple
from loguru import logger
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()
@dataclass
class CoinbaseOrderBook:
    symbol:str;bids:List[Tuple[float,float]];asks:List[Tuple[float,float]];timestamp:float
class CoinbaseConnector:
    def __init__(self,sandbox:bool=False):
        self.sandbox,self.api_key,self.api_secret,self.passphrase=sandbox,os.getenv('COINBASE_API_KEY'),os.getenv('COINBASE_SECRET'),os.getenv('COINBASE_PASSPHRASE')
        self.rest_url="https://api-public.sandbox.pro.coinbase.com"if sandbox else"https://api.pro.coinbase.com"
        self.ws_url="wss://ws-feed-public.sandbox.pro.coinbase.com"if sandbox else"wss://ws-feed.pro.coinbase.com"
        self.exchange,self.session,self.connected=None,None,False
        self.rate_limits={'private':10,'public':10};self.last_request_times={'private':0,'public':0}
        self.ws_subscriptions,self.ws_connection=set(),None
    async def connect(self):
        try:
            self.exchange=ccxt.coinbasepro({'apiKey':self.api_key,'secret':self.api_secret,'password':self.passphrase,'sandbox':self.sandbox,'enableRateLimit':True})
            self.session=aiohttp.ClientSession();await self.exchange.load_markets();self.connected=True
            if self.api_key:
                try:accounts=await self.exchange.fetch_balance();logger.info(f"✅ Coinbase authenticated - {len(accounts.get('info',[]))} accounts")
                except Exception as e:logger.warning(f"⚠️ Auth test failed: {e}")
            logger.info("✅ Connected to Coinbase Pro");asyncio.create_task(self._maintain_websocket())
        except Exception as e:logger.error(f"❌ Coinbase connection failed: {e}");self.connected=False;raise
    async def _rate_limit(self,endpoint_type:str):
        current_time,last_request,min_interval=time.time(),self.last_request_times[endpoint_type],1.0/self.rate_limits[endpoint_type]
        if(time_since_last:=current_time-last_request)<min_interval:await asyncio.sleep(min_interval-time_since_last)
        self.last_request_times[endpoint_type]=time.time()
    async def get_crypto_pairs(self)->List[str]:
        await self._rate_limit('public')
        async with self.session.get(f"{self.rest_url}/products")as response:
            if response.status==200:
                products,crypto_pairs=await response.json(),[]
                for product in products:
                    if product.get('status')=='online'and not product.get('trading_disabled',False):
                        base,quote=product['base_currency'],product['quote_currency']
                        if base in['BTC','ETH','ADA','SOL','DOT','MATIC','AVAX','LINK','UNI']and quote in['USD','USDC','BTC','ETH']:crypto_pairs.append(product['id'])
                return crypto_pairs[:20]
            return[]
    async def fetch_ticker(self,symbol:str)->Dict:
        await self._rate_limit('public');return await self.exchange.fetch_ticker(symbol)
    async def fetch_order_book(self,symbol:str,limit:int=50)->CoinbaseOrderBook:
        await self._rate_limit('public');order_book=await self.exchange.fetch_order_book(symbol,limit)
        return CoinbaseOrderBook(symbol=symbol,bids=[(bid[0],bid[1])for bid in order_book['bids'][:limit]],asks=[(ask[0],ask[1])for ask in order_book['asks'][:limit]],timestamp=time.time())
    async def create_order(self,symbol:str,order_type:str,side:str,amount:float,price:float=None,params:Dict=None)->Dict:
        if not self.api_key:raise ValueError("API credentials required")
        await self._rate_limit('private');params=params or{};params.update({'time_in_force':'GTC','post_only':False})
        order=await self.exchange.create_order(symbol=symbol,type=order_type,side=side,amount=amount,price=price,params=params)
        logger.info(f"📝 Order created: {symbol} {side} {amount} @ {price or'market'}");return order
    async def fetch_balance(self)->Dict:
        if not self.api_key:return{}
        await self._rate_limit('private');balance=await self.exchange.fetch_balance();balances={}
        for currency,info in balance.items():
            if isinstance(info,dict)and info.get('total',0)>0:balances[currency]={'total':info['total'],'free':info['free'],'used':info['used']}
        return balances
    async def _maintain_websocket(self):
        while self.connected:
            try:
                async with websockets.connect(self.ws_url)as websocket:
                    logger.info("🔗 Coinbase WebSocket connected");self.ws_connection=websocket
                    subscribe_msg={"type":"subscribe","channels":[{"name":"ticker","product_ids":await self.get_crypto_pairs()}]}
                    await websocket.send(json.dumps(subscribe_msg))
                    async for message in websocket:
                        data=json.loads(message)
                        if data.get('type')=='ticker':logger.debug(f"📊 {data['product_id']}: ${float(data['price']):.2f}")
            except Exception as e:logger.warning(f"🔄 WebSocket disconnected: {e}");await asyncio.sleep(5)
    async def disconnect(self):
        self.connected=False
        if self.ws_connection:await self.ws_connection.close()
        if self.session:await self.session.close()
        if self.exchange:await self.exchange.close()