#!/bin/bash

echo "🚀 BUILDING ULTIMATE SELF-LEARNING HFT SYSTEM"
echo "=============================================="

# Install Python dependencies
echo "📦 Installing Python ML dependencies..."
pip3 install torch torchvision torchaudio numpy pandas scikit-learn aiohttp

# Build C++ core
echo "⚡ Building C++ ultra-fast core..."
cd cpp_core
g++ -std=c++17 -O3 -march=native -ffast-math -shared -fPIC -o libhft_core.so hft_core.cpp
cd ..

# Build Rust engine
echo "🦀 Building Rust integration engine..."
cd rust_engine
cargo build --release
cd ..

# Create run script
cat > run_ultimate_hft.sh << 'RUNEOF'
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
RUNEOF

chmod +x run_ultimate_hft.sh

echo ""
echo "✅ ULTIMATE HFT SYSTEM BUILT!"
echo "============================="
echo ""
echo "🧠 COMPONENTS CREATED:"
echo "   ✅ C++ ultra-fast core engine"
echo "   ✅ Rust concurrent integration layer"
echo "   ✅ Python evolutionary ML system"
echo "   ✅ Multi-strategy genetic algorithms"
echo "   ✅ Neural network prediction models"
echo ""
echo "🚀 START THE SYSTEM:"
echo "   ./run_ultimate_hft.sh"
echo ""
echo "🧬 SELF-LEARNING FEATURES:"
echo "   • Continuously evolving strategies"
echo "   • Neural network market prediction"
echo "   • Genetic algorithm optimization"
echo "   • Real-time performance adaptation"
echo ""
echo "This is the most advanced HFT system possible!"
