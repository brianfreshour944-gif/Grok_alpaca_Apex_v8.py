import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import logging
from IPython.display import clear_output

from ml_predictor import MLPredictor, GrokGQA_Transformer # Import the necessary classes

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(
        self,
        predictor: MLPredictor,
        historical_df: pd.DataFrame,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001, # 0.1% transaction cost
        prediction_threshold: float = 0.6, # Probability threshold for a 'buy' signal
        seq_len: int = 5 # Must match the seq_len used during training and in MLPredictor
    ):
        super().__init__() # Call parent constructor
        self.predictor = predictor
        self.historical_df = historical_df.copy()
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.prediction_threshold = prediction_threshold
        self.seq_len = seq_len

        self.capital = initial_capital
        self.position = 0 # 0 for no position, 1 for long
        self.entry_price = 0.0 # To track individual trade profit/loss
        self.equity_curve = []
        self.trades = []
        self.total_wins = 0
        self.total_losses = 0
        self.profits = []
        self.losses = []

        logger.info(f"Backtest Engine Initialized with:\n"\
                    f"  Initial Capital: {initial_capital}\n"\
                    f"  Transaction Cost: {transaction_cost * 100:.2f}%\n"\
                    f"  Prediction Threshold: {prediction_threshold}\n"\
                    f"  Sequence Length: {seq_len}")

    def run_backtest(self):
        logger.info("Starting backtest.")
        # Ensure enough data for initial sequence and a subsequent trade
        if len(self.historical_df) < self.seq_len + 1:
            logger.error(f"Not enough historical data for backtest. Requires at least {self.seq_len + 1} data points.")
            return

        for i in range(self.seq_len, len(self.historical_df)):
            current_time = self.historical_df.index[i]
            # Take the last `seq_len` data points for prediction
            sample_df = self.historical_df.iloc[i-self.seq_len : i]

            if len(sample_df) < self.seq_len:
                # This case should ideally not happen if loop starts from self.seq_len
                # but added as a safeguard.
                self.equity_curve.append(self.capital) # Maintain current equity if not enough data
                continue

            prediction = self.predictor.predict(sample_df)

            current_close_price = self.historical_df['close'].iloc[i]
            current_equity_value = self.capital
            if self.position > 0: # If holding units
                current_equity_value += self.position * current_close_price # cash + value of units held
            self.equity_curve.append(current_equity_value)

            if prediction > self.prediction_threshold and self.position == 0: # Buy signal and no position
                # Buy as much as possible with current capital
                units_to_buy = (self.capital * (1 - self.transaction_cost)) / current_close_price
                self.position = units_to_buy
                self.capital = 0 # All cash converted to asset (minus transaction cost)
                self.entry_price = current_close_price
                self.trades.append({
                    'time': current_time,
                    'type': 'BUY',
                    'price': current_close_price,
                    'units': units_to_buy,
                    'equity': current_equity_value
                })
                logger.info(f"[{current_time}] BUY @ {current_close_price:.2f} (Pred: {prediction:.2f})")
            elif prediction < (1 - self.prediction_threshold) and self.position > 0: # Sell signal and holding position
                # Sell all units
                trade_profit_loss = (current_close_price - self.entry_price) * self.position - (self.position * current_close_price * self.transaction_cost)
                if trade_profit_loss > 0:
                    self.total_wins += 1
                    self.profits.append(trade_profit_loss)
                else:
                    self.total_losses += 1
                    self.losses.append(trade_profit_loss)

                self.capital = self.position * current_close_price * (1 - self.transaction_cost) # Cash from selling (minus transaction cost)
                self.trades.append({
                    'time': current_time,
                    'type': 'SELL',
                    'price': current_close_price,
                    'units': self.position,
                    'equity': current_equity_value
                })
                self.position = 0 # No position
                self.entry_price = 0.0 # Reset entry price
                logger.info(f"[{current_time}] SELL @ {current_close_price:.2f} (Pred: {prediction:.2f})")

        # Ensure final equity includes any open positions
        final_equity = self.capital
        if self.position > 0:
            final_equity += self.position * self.historical_df['close'].iloc[-1]
            # Calculate profit/loss for the last open position if any
            trade_profit_loss = (self.historical_df['close'].iloc[-1] - self.entry_price) * self.position - (self.position * self.historical_df['close'].iloc[-1] * self.transaction_cost)
            if trade_profit_loss > 0:
                self.total_wins += 1
                self.profits.append(trade_profit_loss)
            else:
                self.total_losses += 1
                self.losses.append(trade_profit_loss)

        # Only update the last element if equity_curve is not empty
        if self.equity_curve:
            self.equity_curve[-1] = final_equity # Update last point with final equity
        else: # Handle case where backtest didn't run long enough to populate equity_curve
            self.equity_curve.append(final_equity)

        logger.info("Backtest finished.")

    def plot_equity_curve(self):
        if not self.equity_curve:
            logger.warning("No equity curve data to plot. Run backtest first.")
            return

        clear_output(wait=True)
        plt.figure(figsize=(12, 6))
        plt.plot(self.historical_df.index[self.seq_len : len(self.historical_df)], self.equity_curve)
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        plt.show()

    def calculate_metrics(self):
        if not self.equity_curve:
            logger.warning("No equity curve data to calculate metrics. Run backtest first.")
            return {}

        equity = pd.Series(self.equity_curve)
        initial = self.initial_capital
        final = equity.iloc[-1]

        total_return = (final - initial) / initial

        # Daily returns for Sharpe Ratio
        returns = equity.pct_change().dropna() # Changed to equity returns from initial to final
        if len(returns) > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252*24) # Assuming hourly data, 252 trading days per year, 24 hours/day
        else:
            sharpe_ratio = np.nan

        # Max Drawdown
        peak = equity.expanding(min_periods=1).max()
        drawdown = (equity - peak) / peak
        max_drawdown = drawdown.min()

        win_rate = self.total_wins / (self.total_wins + self.total_losses) if (self.total_wins + self.total_losses) > 0 else 0
        avg_profit_per_win = np.mean(self.profits) if self.profits else 0
        avg_loss_per_loss = np.mean(self.losses) if self.losses else 0

        metrics = {
            'Initial Capital': initial,
            'Final Capital': final,
            'Total Return': f'{total_return:.2%}',
            'Sharpe Ratio (Annualized)': f'{sharpe_ratio:.2f}' if not np.isnan(sharpe_ratio) else 'N/A',
            'Max Drawdown': f'{max_drawdown:.2%}',
            'Number of Trades': len([t for t in self.trades if t['type'] == 'BUY']),
            'Total Wins': self.total_wins,
            'Total Losses': self.total_losses,
            'Win Rate': f'{win_rate:.2%}',
            'Average Profit per Win': f'{avg_profit_per_win:.2f}',
            'Average Loss per Loss': f'{avg_loss_per_loss:.2f}'
        }

        print("--- Backtest Metrics ---")
        for k, v in metrics.items():
            print(f"{k}: {v}")
        print("------------------------")

        return metrics
