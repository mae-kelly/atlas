#!/usr/bin/env python3
"""
Check which packages are available and create compatibility layer
"""
import sys
import importlib

def check_package(package_name):
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

# Check critical packages
packages = [
    'numpy', 'pandas', 'matplotlib', 'aiohttp', 'requests',
    'torch', 'sklearn', 'scipy', 'yfinance', 'ccxt',
    'textblob', 'vaderSentiment', 'feedparser', 'loguru'
]

print("📦 Package Availability Check")
print("=" * 40)

available = []
missing = []

for package in packages:
    if check_package(package):
        print(f"✅ {package}")
        available.append(package)
    else:
        print(f"❌ {package}")
        missing.append(package)

print(f"\n📊 Summary: {len(available)}/{len(packages)} packages available")

if missing:
    print(f"\n⚠️  Missing packages: {', '.join(missing)}")
    print("\nTo install missing packages:")
    for pkg in missing:
        if pkg == 'torch':
            print(f"pip install torch --index-url https://download.pytorch.org/whl/cpu")
        elif pkg == 'sklearn':
            print(f"pip install scikit-learn")
        else:
            print(f"pip install {pkg}")

print(f"\n✅ System is {'ready' if len(missing) <= 2 else 'partially ready'} for AI Trading Empire")
