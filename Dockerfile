# We are using 3.10 because it has the best compatibility for pandas-ta
FROM python:3.10-slim

WORKDIR /app

# Install basic tools the server needs to build trading libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# We install them in this specific order to avoid the version clash
RUN pip install --no-cache-dir ccxt pandas numpy tqdm
RUN pip install --no-cache-dir pandas_ta

CMD ["python3", "Grok_OKX_Apex_v8.py"]
