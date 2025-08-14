#!/bin/bash

echo "🚀 DEPLOYING 24/7 AUTONOMOUS TRADING SYSTEM"
echo "==========================================="

# Build the Docker image
echo "🔨 Building Docker image..."
docker build -t autonomous_trader .

# Create data directories
mkdir -p data logs

# Set permissions
chmod 755 data logs

# Start with Docker Compose
echo "🐳 Starting with Docker Compose..."
docker-compose up -d

# Wait for startup
echo "⏳ Waiting for startup..."
sleep 10

# Check status
if docker ps | grep -q "autonomous_trader"; then
    echo "✅ Autonomous trader is running!"
    echo ""
    echo "📊 Access monitoring:"
    echo "   • Health check: http://localhost:8080/health"
    echo "   • Logs: docker logs autonomous_trader -f"
    echo "   • Monitor: ./monitor.sh"
    echo ""
    echo "🛡️  Safety features:"
    echo "   • Protected balance: $500 (never goes below)"
    echo "   • Maximum daily risk: 10%"
    echo "   • Maximum risk per trade: 2%"
    echo "   • Automatic restart on failure"
    echo "   • Health monitoring"
    echo ""
    echo "🛑 To stop: docker-compose down"
else
    echo "❌ Failed to start autonomous trader"
    docker logs autonomous_trader
fi
