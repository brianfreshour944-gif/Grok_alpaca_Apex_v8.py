FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1
RUN pip install ccxt pandas numpy torch pandas_ta tqdm --index-url https://download.pytorch.org/whl/cpu
WORKDIR /app
COPY . .
# We run the bot by default
CMD ["python", "ultimate_bot.py"]
