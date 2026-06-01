# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Prevent interactive prompts and set the working directory
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Copy requirements file first to take advantage of Docker caching
COPY requirements.txt .

# Install all dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project code
COPY . .

# Run your application
CMD ["python", "train_transformer.py"]
