#!/bin/bash

echo "🔍 CHECKING SYSTEM REQUIREMENTS"
echo "==============================="

# Check Docker
if command -v docker &> /dev/null; then
    echo "✅ Docker is installed"
    docker --version
else
    echo "❌ Docker is not installed"
    echo "   Install with: curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh"
fi

# Check Docker Compose
if command -v docker-compose &> /dev/null; then
    echo "✅ Docker Compose is installed"
    docker-compose --version
else
    echo "❌ Docker Compose is not installed"
    echo "   Install with: sudo apt-get install docker-compose"
fi

# Check Rust
if command -v cargo &> /dev/null; then
    echo "✅ Rust is installed"
    cargo --version
else
    echo "❌ Rust is not installed"
    echo "   Install with: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
fi

# Check Python
if command -v python3 &> /dev/null; then
    echo "✅ Python 3 is installed"
    python3 --version
else
    echo "❌ Python 3 is not installed"
    echo "   Install with: sudo apt-get install python3 python3-pip"
fi

# Check available memory
echo ""
echo "💾 System Resources:"
free -h
echo ""
echo "💻 CPU Info:"
nproc
echo "cores available"

echo ""
echo "🚀 Ready to deploy autonomous trading system!"
