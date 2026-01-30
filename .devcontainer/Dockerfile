FROM python:3.14-slim

# System deps (adjust if you need build tools)
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Make uv available in PATH
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /workspace
