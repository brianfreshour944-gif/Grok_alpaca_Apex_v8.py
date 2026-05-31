FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir pandas numpy alpaca-py asyncio

COPY . .

# Change this to the actual main file you want to execute
CMD ["python", "ml_predictor.py"]
