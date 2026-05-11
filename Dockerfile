# Using 3.11 to satisfy pandas_ta requirements
FROM python:3.11-slim

WORKDIR /app

# Install system tools needed for building some trading libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Install the libraries one by one to ensure they stick
RUN pip install --no-cache-dir ccxt pandas numpy tqdm
RUN pip install --no-cache-dir pandas_ta

CMD ["python3", "Grok_OKX_Apex_v8.py"]
