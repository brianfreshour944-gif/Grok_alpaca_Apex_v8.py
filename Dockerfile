# We use Python 3.10 as it has the best compatibility for pandas-ta
FROM python:3.10-slim

WORKDIR /app

# Upgrade pip to help find the correct library versions
RUN pip install --upgrade pip

# Copy your files into the container
COPY . .

# Install requirements
# We use a specific beta version of pandas-ta that avoids the version clash
RUN pip install --no-cache-dir ccxt pandas numpy tqdm
RUN pip install --no-cache-dir pandas-ta==0.3.14b0

# The command to run your bot
CMD ["python3", "Grok_OKX_Apex_v8.py"]
