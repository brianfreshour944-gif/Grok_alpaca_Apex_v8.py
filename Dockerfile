# We upgraded to 3.11 to satisfy the pandas_ta requirements
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies first
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# We install specific versions to prevent the "No matching distribution" error
RUN pip install --no-cache-dir ccxt pandas numpy pandas_ta tqdm

CMD ["python3", "Grok_OKX_Apex_v8.py"]
