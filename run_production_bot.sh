#!/bin/bash

export RUST_LOG=info
export RUST_BACKTRACE=1

cd crypto_trading

if [ "$1" = "docker" ]; then
    echo "Running in Docker..."
    cd docker
    docker-compose up --build
else
    echo "Running natively..."
    ./target/release/crypto_trading
fi
