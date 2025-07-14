#!/usr/bin/env python3
"""Atlas Benchmark - Compressed Version"""
import sys,os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))

def main():
    try:
        from src.models.benchmark import CompactBenchmark
        benchmark = CompactBenchmark(scale_factor=1)
        return benchmark.run_benchmark()
    except Exception as e:
        print(f"❌ Benchmark error: {e}")
        return None

if __name__ == "__main__":
    main()
