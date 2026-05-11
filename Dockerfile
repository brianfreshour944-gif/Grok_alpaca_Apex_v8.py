FROM python:3.12-slim
WORKDIR /app
RUN pip install --upgrade pip
COPY . .
RUN pip install --no-cache-dir ccxt pandas numpy tqdm pandas-ta
CMD ["python3", "Grok_OKX_Apex_v8.py"]
