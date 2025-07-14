import time,psutil,threading,asyncio
from typing import Dict,List
from collections import deque
from loguru import logger

class PerformanceMonitor:
    def __init__(self,monitoring_interval:float=0.1):
        self.monitoring_interval=monitoring_interval
        self.metrics_history=deque(maxlen=1000)
        self.latency_measurements=deque(maxlen=10000)
        self.throughput_measurements=deque(maxlen=1000)
        self.running=False
        self.performance_targets={'latency_ns':13900,'throughput_msgs_sec':1000000,'cpu_usage_max':80,'memory_usage_max':85}
    
    async def start_monitoring(self):
        self.running=True
        logger.info("📊 Performance monitoring started")
        while self.running:
            try:
                metrics=self._collect_system_metrics()
                self.metrics_history.append(metrics)
                await self._check_performance_alerts(metrics)
                await asyncio.sleep(self.monitoring_interval)
            except Exception as e:logger.error(f"❌ Performance monitoring error: {e}")
    
    def _collect_system_metrics(self)->Dict:
        try:
            cpu_percent=psutil.cpu_percent(interval=None)
            memory=psutil.virtual_memory()
            network=psutil.net_io_counters()
            disk=psutil.disk_io_counters()
            return{'timestamp':time.time(),'cpu_percent':cpu_percent,'memory_percent':memory.percent,'memory_available_gb':memory.available/1024**3,'network_bytes_sent':network.bytes_sent,'network_bytes_recv':network.bytes_recv,'disk_read_mb':disk.read_bytes/1024**2,'disk_write_mb':disk.write_bytes/1024**2}
        except Exception as e:logger.error(f"❌ System metrics collection error: {e}");return{}
    
    async def _check_performance_alerts(self,metrics:Dict):
        alerts=[]
        if metrics.get('cpu_percent',0)>self.performance_targets['cpu_usage_max']:
            alerts.append(f"🚨 CPU usage high: {metrics['cpu_percent']:.1f}%")
        if metrics.get('memory_percent',0)>self.performance_targets['memory_usage_max']:
            alerts.append(f"🚨 Memory usage high: {metrics['memory_percent']:.1f}%")
        if len(self.latency_measurements)>0:
            avg_latency=sum(self.latency_measurements)/len(self.latency_measurements)
            if avg_latency>self.performance_targets['latency_ns']:
                alerts.append(f"🚨 Latency high: {avg_latency:.0f}ns")
        for alert in alerts:logger.warning(alert)
    
    def record_latency(self,latency_ns:int):
        self.latency_measurements.append(latency_ns)
    
    def record_throughput(self,messages_per_second:float):
        self.throughput_measurements.append(messages_per_second)
    
    def get_performance_summary(self)->Dict:
        if not self.metrics_history:return{}
        recent_metrics=list(self.metrics_history)[-10:]
        avg_cpu=sum(m.get('cpu_percent',0)for m in recent_metrics)/len(recent_metrics)
        avg_memory=sum(m.get('memory_percent',0)for m in recent_metrics)/len(recent_metrics)
        avg_latency=sum(self.latency_measurements)/len(self.latency_measurements)if self.latency_measurements else 0
        avg_throughput=sum(self.throughput_measurements)/len(self.throughput_measurements)if self.throughput_measurements else 0
        return{'avg_cpu_percent':avg_cpu,'avg_memory_percent':avg_memory,'avg_latency_ns':avg_latency,'avg_throughput_msgs_sec':avg_throughput,'performance_score':self._calculate_performance_score(avg_cpu,avg_memory,avg_latency,avg_throughput),'targets_met':self._check_targets_met(avg_cpu,avg_memory,avg_latency,avg_throughput)}
    
    def _calculate_performance_score(self,cpu:float,memory:float,latency:float,throughput:float)->float:
        cpu_score=max(0,100-cpu)/100
        memory_score=max(0,100-memory)/100
        latency_score=max(0,min(1,self.performance_targets['latency_ns']/max(latency,1)))
        throughput_score=min(1,throughput/self.performance_targets['throughput_msgs_sec'])
        return(cpu_score+memory_score+latency_score+throughput_score)/4
    
    def _check_targets_met(self,cpu:float,memory:float,latency:float,throughput:float)->Dict:
        return{'cpu_target_met':cpu<=self.performance_targets['cpu_usage_max'],'memory_target_met':memory<=self.performance_targets['memory_usage_max'],'latency_target_met':latency<=self.performance_targets['latency_ns'],'throughput_target_met':throughput>=self.performance_targets['throughput_msgs_sec']}
    
    def stop_monitoring(self):
        self.running=False
        logger.info("⏹️ Performance monitoring stopped")
