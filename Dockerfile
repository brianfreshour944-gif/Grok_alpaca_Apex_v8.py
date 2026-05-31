# Use an official Python image
FROM python:3.11-slim

# Install system dependencies required for heavy ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the container working directory
WORKDIR /app

# Install all data, trading, ML, and technical analysis libraries
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    alpaca-py \
    asyncio \
    joblib \
    scikit-learn \
    pandas_ta \
    torch --extra-index-url https://download.pytorch.org/whl/cpu

# Copy all your project files into the container
COPY . .

# Run your machine learning trading entry point
CMD ["python", "ml_predictor.py"]
