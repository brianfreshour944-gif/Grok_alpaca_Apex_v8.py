# Using 3.12 which is the primary requirement for the latest pandas-ta
FROM python:3.12-slim

WORKDIR /app

# Upgrade pip immediately to ensure it can find the newest library versions
RUN pip install --upgrade pip

COPY . .

# Install libraries in one command to help pip resolve the versions correctly
RUN pip install --no-cache-dir ccxt pandas numpy pandas_ta tqdm

CMD ["python3", "Grok_OKX_Apex_v8.py"]
