import asyncio
import ccxt.pro as ccxtpro
import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
from brokers import GrokOKXBroker
from ml_predictor import MLPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class GrokApexIroncladBot:
    def __init__(self, paper_mode: bool = True):
        self.broker = GrokOKXBroker(paper_mode=paper_mode)
        self.symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
        self.ml = MLPredictor(model_path="grok_gqa_v9_best.pth", seq_len=512)
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.running = True
        self.start_balance = None
        self.load_state()

    def load_state(self):
        if os.path.exists("grok_apex_state.json"):
            with open("grok_apex_state.json") as f:
                data = json.load(f)
                self.positions = data.get("positions", {})
                self.trades = data.get("trades", [])

    def save_state(self):
        with open("grok_apex_state.json", "w") as f:
            json.dump({"positions": self.positions, "trades": self.trades[-100:]}, f)

    async def run(self):
        exchange = ccxtpro.okx({
            'apiKey': self.broker.api_key,
            'secret': self.broker.secret,
            'password': self.broker.passphrase,
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'}
        })
        if self.broker.paper_mode: exchange.set_sandbox_mode(True)

        while self.running:
            balance = self.broker.get_balance('USDT')
            for symbol in self.symbols:
                try:
                    ticker = await exchange.watch_ticker(symbol)
                    ohlcv = await exchange.fetch_ohlcv(symbol, '15m', limit=600)
                    df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
                    
                    # Basic feature engineering for the predictor
                    df['returns'] = df['close'].pct_change()
                    df['vol_14'] = df['returns'].rolling(14).std()
                    # (MLPredictor handles the rest)
                    
                    score = self.ml.predict(df)
                    price = ticker['last']
                    logger.info(f"{symbol} | Price: ${price} | Score: {score:.3f}")

                    if score > 0.67 and symbol not in self.positions:
                        await exchange.create_order(symbol, 'market', 'buy', 0.01)
                        self.positions[symbol] = {'price': price}
                    elif score < 0.36 and symbol in self.positions:
                        await exchange.create_order(symbol, 'market', 'sell', 0.01)
                        del self.positions[symbol]

                except Exception as e:
                    logger.error(f"Error: {e}")
            await asyncio.sleep(15)

if __name__ == "__main__":
    bot = GrokApexIroncladBot(paper_mode=True)
    asyncio.run(bot.run())
