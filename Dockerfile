FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch CPU first on its own (large, needs separate index)
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu

# Install everything else except pandas-ta
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    alpaca-py \
    joblib \
    scikit-learn \
    ccxt

# Install pandas-ta from GitHub (PyPI has no ARM64 wheel)
RUN pip install --no-cache-dir \
    "git+https://github.com/twopirllc/pandas-ta.git@development"

COPY . .

CMD ["python", "Grok_OKX_Apex_v8.py"]
