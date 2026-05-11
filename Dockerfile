# We use 3.9 because pandas_ta 0.4.67b0 specifically supports it best
FROM python:3.9-slim

WORKDIR /app

# Install system tools
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Update pip first to help find the right library versions
RUN pip install --upgrade pip

# Install requirements
RUN pip install --no-cache-dir ccxt pandas numpy tqdm
RUN pip install --no-cache-dir pandas_ta

CMD ["python3", "Grok_OKX_Apex_v8.py"]
