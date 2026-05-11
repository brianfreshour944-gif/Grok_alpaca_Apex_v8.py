# Use Python 3.12 for the best library support
FROM python:3.12-slim

WORKDIR /app

# Upgrade pip to handle the modern technical analysis libraries
RUN pip install --upgrade pip

# Copy your bot code into the container
COPY . .

# Install libraries. We use 'pandas-ta' (with a hyphen) for the installer.
RUN pip install --no-cache-dir ccxt pandas numpy tqdm
RUN pip install --no-cache-dir pandas-ta

# Run your specific script
CMD ["python3", "Grok_OKX_Apex_v8.py"]
