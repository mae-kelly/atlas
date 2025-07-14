#!/bin/bash

set -e

echo "🔧 Setting up high-performance infrastructure"

sysctl -w net.core.rmem_max=268435456
sysctl -w net.core.wmem_max=268435456
sysctl -w net.ipv4.tcp_rmem="4096 65536 268435456"
sysctl -w net.ipv4.tcp_wmem="4096 65536 268435456"
sysctl -w net.core.netdev_max_backlog=5000
sysctl -w net.ipv4.tcp_congestion_control=bbr

echo 'performance' | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

if [ -f /proc/sys/kernel/sched_rt_runtime_us ]; then
    echo -1 > /proc/sys/kernel/sched_rt_runtime_us
fi

echo 0 > /proc/sys/kernel/numa_balancing

taskset -c 0 python3 -c "
import os
os.sched_setaffinity(0, {0})
print('CPU affinity set for core 0')
"

ulimit -n 1048576
echo "vm.swappiness=1" >> /etc/sysctl.conf
echo "kernel.shmmax=68719476736" >> /etc/sysctl.conf

mkdir -p /dev/hugepages
mount -t hugetlbfs nodev /dev/hugepages
echo 1024 > /proc/sys/vm/nr_hugepages

echo "✅ High-performance infrastructure configured"
