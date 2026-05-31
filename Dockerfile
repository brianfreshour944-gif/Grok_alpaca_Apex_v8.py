# Use an official Python image
FROM python:3.11-slim

# Install system dependencies required for heavy ML libraries like PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the container working directory
WORKDIR /app

# Install your baseline trading libraries AND PyTorch (CPU version for server efficiency)
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    alpaca-py \
    asyncio \
    torch --extra-index-url https://download.pytorch.org/whl/cpu

# Copy all your project files into the container
COPY . .

# Run your machine learning trading entry point
CMD ["python", "ml_predictor.py"]
