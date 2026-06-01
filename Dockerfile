# Use a slim Python 3.11 base image
FROM python:3.11-slim

# Set non-interactive environment to prevent the debconf errors you saw
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Copy only the requirements file first (Best Practice for Docker caching)
COPY requirements.txt .

# Install dependencies in one layer
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Run the entrypoint script
CMD ["python", "train_transformer.py"]
