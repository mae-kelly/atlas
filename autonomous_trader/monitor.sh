#!/bin/bash

echo "📊 AUTONOMOUS TRADER MONITORING"
echo "==============================="

while true; do
    echo "$(date): Checking trader status..."
    
    # Check if container is running
    if docker ps | grep -q "autonomous_trader"; then
        echo "✅ Container is running"
        
        # Check health endpoint
        if curl -s http://localhost:8080/health | grep -q "OK"; then
            echo "✅ Health check passed"
        else
            echo "⚠️  Health check failed - restarting container"
            docker restart autonomous_trader
        fi
        
        # Show recent logs
        echo "📝 Recent activity:"
        docker logs autonomous_trader --tail 5
        
    else
        echo "❌ Container not running - starting..."
        docker-compose up -d
    fi
    
    echo "---"
    sleep 60  # Check every minute
done
