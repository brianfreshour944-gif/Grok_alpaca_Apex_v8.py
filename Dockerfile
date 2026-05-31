# Use an official Python image
FROM python:3.11-slim

# Install system dependencies required for compiling modules on ARM64
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the container working directory
WORKDIR /app

# Install all data, trading, and ML libraries in a solid single-line block to avoid parsing errors
RUN pip install --no-cache-dir pandas numpy alpaca-py asyncio joblib scikit-learn git+https://github.com/twopirllc/pandas-ta.git@development torch --extra-index-url https://download.pytorch.org/whl/cpu

# Copy all your project files into the container
COPY . .

# Run your machine learning trading entry point
CMD ["python", "ml_predictor.py"]
