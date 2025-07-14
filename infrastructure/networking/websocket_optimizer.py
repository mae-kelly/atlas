import asyncio,websockets,aiohttp,json,time,zlib
from typing import Dict,List,Callable
import concurrent.futures
from loguru import logger

class OptimizedWebSocketManager:
    def __init__(self,max_connections:int=1024,compression_level:int=6):
        self.max_connections,self.compression_level=max_connections,compression_level
        self.connection_pools={}
        self.message_queues={}
        self.compression_enabled=True
        self.binary_encoding=True
    
    async def create_connection_pool(self,exchange:str,endpoints:List[str])->Dict:
        self.connection_pools[exchange]=[]
        self.message_queues[exchange]=asyncio.Queue(maxsize=10000)
        connections_created=0
        for endpoint in endpoints:
            try:
                for i in range(min(self.max_connections//len(endpoints),256)):
                    ws=await websockets.connect(endpoint,compression='deflate'if self.compression_enabled else None,max_size=2**23,max_queue=1000)
                    self.connection_pools[exchange].append({'websocket':ws,'endpoint':endpoint,'created_at':time.time(),'messages_received':0})
                    connections_created+=1
            except Exception as e:logger.error(f"❌ Connection pool creation failed for {endpoint}: {e}")
        logger.info(f"🔗 Created {connections_created} connections for {exchange}")
        return{'connections_created':connections_created,'compression_enabled':self.compression_enabled,'max_connections':self.max_connections}
    
    async def optimized_message_handler(self,exchange:str,message_callback:Callable):
        while True:
            try:
                if exchange in self.connection_pools:
                    tasks=[]
                    for conn_info in self.connection_pools[exchange]:
                        task=asyncio.create_task(self._handle_connection_messages(conn_info,message_callback))
                        tasks.append(task)
                    if tasks:await asyncio.gather(*tasks,return_exceptions=True)
            except Exception as e:logger.error(f"❌ Message handler error for {exchange}: {e}")
            await asyncio.sleep(0.001)
    
    async def _handle_connection_messages(self,conn_info:Dict,callback:Callable):
        try:
            ws=conn_info['websocket']
            async for message in ws:
                start_time=time.perf_counter_ns()
                if self.binary_encoding and isinstance(message,bytes):
                    decoded_message=self._decode_binary_message(message)
                else:decoded_message=json.loads(message)if isinstance(message,str)else message
                await callback(decoded_message)
                conn_info['messages_received']+=1
                processing_time_ns=time.perf_counter_ns()-start_time
                if processing_time_ns>1000000:logger.warning(f"⚠️ Slow message processing: {processing_time_ns/1000000:.2f}ms")
        except websockets.exceptions.ConnectionClosed:logger.warning(f"🔌 WebSocket connection closed for {conn_info['endpoint']}")
        except Exception as e:logger.error(f"❌ Connection message handling error: {e}")
    
    def _decode_binary_message(self,message:bytes)->Dict:
        try:
            if self.compression_enabled:message=zlib.decompress(message)
            return json.loads(message.decode('utf-8'))
        except:return{}
    
    def get_connection_stats(self)->Dict:
        stats={}
        for exchange,connections in self.connection_pools.items():
            total_messages=sum(conn['messages_received']for conn in connections)
            active_connections=len([conn for conn in connections if time.time()-conn['created_at']<3600])
            stats[exchange]={'total_connections':len(connections),'active_connections':active_connections,'total_messages_received':total_messages,'avg_messages_per_connection':total_messages/max(len(connections),1)}
        return stats

class ChronicleQueueIntegration:
    def __init__(self,queue_path:str='/tmp/trading_queue'):
        self.queue_path,self.queue_initialized=queue_path,False
        self.message_count,self.throughput_target=0,1000000
    
    def initialize_queue(self)->bool:
        try:
            import os;os.makedirs(self.queue_path,exist_ok=True)
            self.queue_initialized=True
            logger.info(f"📁 Chronicle Queue initialized at {self.queue_path}")
            return True
        except Exception as e:logger.error(f"❌ Chronicle Queue initialization failed: {e}");return False
    
    def write_message(self,message:Dict)->bool:
        if not self.queue_initialized:return False
        try:
            timestamp_ns=time.perf_counter_ns()
            message_with_timestamp={'timestamp_ns':timestamp_ns,'data':message}
            with open(f"{self.queue_path}/messages_{int(time.time())}.json",'a')as f:
                f.write(json.dumps(message_with_timestamp)+'\n')
            self.message_count+=1;return True
        except Exception as e:logger.error(f"❌ Message write failed: {e}");return False
    
    def get_throughput_stats(self)->Dict:
        return{'messages_written':self.message_count,'target_throughput':self.throughput_target,'queue_path':self.queue_path,'initialized':self.queue_initialized}

class MulticastDataDistribution:
    def __init__(self,multicast_group:str='224.1.1.1',port:int=5007):
        self.multicast_group,self.port=multicast_group,port
        self.subscribers,self.message_count=[],0
    
    async def start_multicast_server(self,data_callback:Callable):
        try:
            import socket
            sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM,socket.IPPROTO_UDP)
            sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            sock.bind(('',self.port))
            mreq=socket.inet_aton(self.multicast_group)+socket.inet_aton('0.0.0.0')
            sock.setsockopt(socket.IPPROTO_IP,socket.IP_ADD_MEMBERSHIP,mreq)
            logger.info(f"📡 Multicast server started on {self.multicast_group}:{self.port}")
            while True:
                data,addr=sock.recvfrom(4096)
                try:
                    message=json.loads(data.decode('utf-8'))
                    await data_callback(message)
                    self.message_count+=1
                except Exception as e:logger.error(f"❌ Multicast message processing error: {e}")
        except Exception as e:logger.error(f"❌ Multicast server error: {e}")
    
    def broadcast_message(self,message:Dict)->bool:
        try:
            import socket
            sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM,socket.IPPROTO_UDP)
            sock.setsockopt(socket.IPPROTO_IP,socket.IP_MULTICAST_TTL,2)
            data=json.dumps(message).encode('utf-8')
            sock.sendto(data,(self.multicast_group,self.port))
            sock.close();return True
        except Exception as e:logger.error(f"❌ Multicast broadcast failed: {e}");return False
