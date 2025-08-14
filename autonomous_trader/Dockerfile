FROM rust:1.75-slim as builder

# Install dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
RUN pip3 install aiohttp requests beautifulsoup4 scikit-learn transformers tokenizers

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY ml_analysis ./ml_analysis

# Build the application
RUN cargo build --release

# Runtime image
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install aiohttp requests beautifulsoup4 scikit-learn transformers tokenizers

WORKDIR /app

# Copy the binary and ML scripts
COPY --from=builder /app/target/release/autonomous_trader .
COPY --from=builder /app/ml_analysis ./ml_analysis

# Create data and logs directories
RUN mkdir -p /app/data /app/logs

# Set permissions
RUN chmod +x autonomous_trader
RUN chmod +x ml_analysis/*.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Run the bot
CMD ["./autonomous_trader"]
