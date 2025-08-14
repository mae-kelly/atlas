#!/bin/bash

echo "Crypto Trading Bot Monitor"
echo "========================="

while true; do
    clear
    echo "$(date)"
    echo ""
    
    # Service status
    echo "Service Status:"
    sudo systemctl is-active crypto-trading-bot
    echo ""
    
    # Recent logs
    echo "Recent Activity:"
    sudo journalctl -u crypto-trading-bot --no-pager -n 20 | tail -10
    echo ""
    
    # System resources
    echo "Resource Usage:"
    echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')"
    echo "Memory: $(free -h | awk 'NR==2{printf "%.1f%%", $3*100/$2 }')"
    echo ""
    
    echo "Press Ctrl+C to exit"
    sleep 10
done
