FROM ghcr.io/ukewea/python-talib:ubuntu24.04-python3.12
# or try: techtrader/python-talib if the above doesn't work

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# === Reliable TA-Lib installation ===
RUN wget https://github.com/TA-Lib/ta-lib/releases/download/v0.4.0/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib-0.4.0 && \
    ./configure --prefix=/usr && \
    make -j2 && \
    make install && \
    cd .. && \
    rm -rf ta-lib-0.4.0* && \
    ldconfig

# Python packages
RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "Grok_OKX_Apex_v8.py"]
