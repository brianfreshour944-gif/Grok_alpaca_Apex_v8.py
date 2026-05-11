# We use Python 3.12 because your logs show pandas-ta requires it
FROM python:3.12-slim

WORKDIR /app

# Update pip to ensure smooth library installation
RUN pip install --upgrade pip

# Copy your bot files
COPY . .

# Install the libraries. 
RUN pip install --no-cache-dir ccxt pandas numpy tqdm pandas-ta

# Run your bot script
CMD ["python3", "Grok_OKX_Apex_v8.py"]
