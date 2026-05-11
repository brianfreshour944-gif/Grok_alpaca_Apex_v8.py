import os
import ccxt

class GrokOKXBroker:
    def __init__(self, paper_mode=True):
        self.paper_mode = paper_mode
        # Pulling from Coolify Environment Variables
        self.api_key = os.getenv('OKX_API_KEY')
        self.secret = os.getenv('OKX_SECRET_KEY')
        self.passphrase = os.getenv('OKX_PASSPHRASE')

    def get_balance(self, asset='USDT'):
        try:
            exchange = ccxt.okx({
                'apiKey': self.api_key,
                'secret': self.secret,
                'password': self.passphrase
            })
            if self.paper_mode: exchange.set_sandbox_mode(True)
            balance = exchange.fetch_balance()
            return balance.get(asset, {}).get('free', 0.0)
        except:
            return 1000.0 # Default for paper testing if keys fail
