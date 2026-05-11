FROM techtrader/python-ta-lib:3.11

WORKDIR /app

RUN apt-get update && apt-get install -y python3-pip

RUN pip install --upgrade pip --break-system-packages

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

COPY . .

CMD ["python3", "Grok_OKX_Apex_v8.py"]
