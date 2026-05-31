FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    pandas \
    numpy \
    alpaca-py \
    joblib \
    scikit-learn \
    pandas-ta \
    torch --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

CMD ["python", "Grok_OKX_Apex_v8.py"]
