#!/bin/bash

echo "🧠 ULTIMATE SELF-LEARNING HFT SYSTEM"
echo "===================================="
echo ""
echo "🏗️  ARCHITECTURE:"
echo "   • C++ Core: Ultra-fast signal processing"
echo "   • Rust Engine: Safe concurrent execution"
echo "   • Python ML: Evolutionary neural networks"
echo "   • Real-time adaptation and learning"
echo ""
echo "🧬 EVOLUTION FEATURES:"
echo "   • Genetic algorithm strategy optimization"
echo "   • Neural network prediction models"
echo "   • Continuous parameter adaptation"
echo "   • Multi-strategy weight evolution"
echo ""
echo "⚡ PERFORMANCE:"
echo "   • 40 Hz market data processing"
echo "   • 100 Hz signal generation"
echo "   • ML evolution every 30 seconds"
echo "   • Real-time strategy weight adaptation"
echo ""

read -p "🚨 Start ultimate self-learning HFT? (y/N): " confirm

if [[ $confirm != [yY] ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "🚀 LAUNCHING ULTIMATE HFT SYSTEM..."

# Set library path for C++
export LD_LIBRARY_PATH=$PWD/cpp_core:$LD_LIBRARY_PATH

# Start the Rust engine (which coordinates everything)
cd rust_engine
./target/release/hft_engine
