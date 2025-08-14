FROM rust:1.75-slim as builder

RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    build-essential \
    cmake \
    clang \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch torchvision torchaudio numpy pandas aiohttp

WORKDIR /app

COPY --from=builder /app/target/release/crypto_trading /app/
COPY --from=builder /app/ml_engine /app/ml_engine/

ENV RUST_LOG=info
ENV RUST_BACKTRACE=1

CMD ["./crypto_trading"]
