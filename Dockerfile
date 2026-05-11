# Use Python 3.12 to meet the requirements of the latest pandas-ta
FROM python:3.12-slim

WORKDIR /app

# Upgrade pip immediately to handle modern library versions
RUN pip install --upgrade pip

# Copy your bot files into the container
COPY . .

# Install libraries in one command to help pip resolve versions correctly
RUN pip install --no-cache-dir ccxt pandas numpy pandas_ta tqdm

# The command to start your bot
CMD ["python3", "Grok_OKX_Apex_v8.py"]
