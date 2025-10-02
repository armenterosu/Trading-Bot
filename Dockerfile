FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ../trading-bot .

ENV PYTHONUNBUFFERED=1

CMD ["python", "src/main.py", "--mode", "backtest", "--config", "config.yaml", "--exchange", "binance", "--symbol", "BTC/USDT", "--timeframe", "1h"]
