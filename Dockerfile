# Use an official lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all your bot files into the container
COPY . /app/

# Install the required trading and data libraries
RUN pip install --no-cache-dir pandas numpy alpaca-py asyncio

# Run your main bot script when the container boots up
CMD ["python", "Grok_OKX_Apex_v8.py"]
