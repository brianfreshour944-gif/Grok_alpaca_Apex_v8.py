# =============================================
# Optimized Dockerfile for Grok_OKX_Apex_v8.py
# =============================================

FROM python:3.12-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies + TA-Lib build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Build and install TA-Lib (C library) from source
RUN wget https://github.com/TA-Lib/ta-lib/releases/download/v0.4.0/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib-0.4.0 && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib-0.4.0 ta-lib-0.4.0-src.tar.gz

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements first (better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application
COPY . .

# Run the bot
CMD ["python3", "Grok_OKX_Apex_v8.py"]
