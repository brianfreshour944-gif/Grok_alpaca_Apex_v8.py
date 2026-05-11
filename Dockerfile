FROM techtrader/python-ta-lib:3.11

WORKDIR /app

# Fix for Debian's externally-managed Python + install pip if needed
RUN apt-get update && apt-get install -y python3-pip && \
    pip install --upgrade pip --break-system-packages

# Copy and install your Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

# Copy your bot code
COPY . .

# Start the bot
CMD ["python3", "Grok_OKX_Apex_v8.py"]
