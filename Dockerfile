
# Use Python 3.12 for library compatibility
FROM python:3.12-slim

WORKDIR /app

# Upgrade pip to handle the modern libraries
RUN pip install --upgrade pip

# Copy your bot files
COPY . .

# Install the libraries one by one
RUN pip install --no-cache-dir ccxt pandas numpy tqdm
RUN pip install --no-cache-dir pandas-ta

# Run your specific bot script
CMD ["python3", "Grok_OKX_Apex_v8.py"]
