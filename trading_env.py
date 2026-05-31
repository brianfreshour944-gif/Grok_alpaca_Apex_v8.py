import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging

# Import the centralized feature engineering function
from feature_engineering import add_features, FEATURE_COLS

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    """Custom Trading Environment that follows gymnasium interface."""

    metadata = {'render_modes': ['human'], 'render_fps': 3}

    def __init__(
        self,
        historical_df: pd.DataFrame,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001,
        seq_len: int = 5,
        reward_type: str = 'portfolio_return',
    ):
        super().__init__()

        self.historical_df = historical_df.copy()
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.seq_len = seq_len
        self.reward_type = reward_type

        self.action_space = spaces.Discrete(3)

        # Define observation space using the actual number of features
        num_features = len(FEATURE_COLS)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.seq_len, num_features), dtype=np.float32
        )

        self._current_step = None
        self.current_capital = initial_capital
        self.current_position = 0
        self.portfolio_value = initial_capital
        self.trades = []
        self.equity_curve = []

        logger.info("Trading Environment Initialized.")

    def _get_observation(self):
        end_index = self._current_step + 1
        start_index = max(0, end_index - self.seq_len)

        obs_df = self.historical_df.iloc[start_index:end_index].copy()

        # Use the centralized feature engineering function
        observation = add_features(obs_df).values

        # Pad with zeros if not enough data points for the sequence length
        if len(observation) < self.seq_len:
            padding = np.zeros((self.seq_len - len(observation), observation.shape[1]))
            observation = np.vstack([padding, observation])

        return observation.astype(np.float32)

    def _get_info(self):
        return {
            "current_capital": self.current_capital,
            "current_position": self.current_position,
            "portfolio_value": self.portfolio_value,
            "current_price": self.historical_df['close'].iloc[self._current_step]
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_step = self.seq_len -1
        self.current_capital = self.initial_capital
        self.current_position = 0
        self.portfolio_value = self.initial_capital
        self.trades = []
        self.equity_curve = [self.initial_capital]

        observation = self._get_observation()
        info = self._get_info()
        logger.info("Environment reset.")
        return observation, info

    def step(self, action):
        self._current_step += 1

        if self._current_step >= len(self.historical_df): # End of episode
            return (
                self._get_observation(),
                self._calculate_reward(),
                True,
                False,
                self._get_info()
            )

        current_price = self.historical_df['close'].iloc[self._current_step]
        reward = 0.0

        if action == 0:  # BUY
            if self.current_position == 0:
                units_to_buy = (self.current_capital * (1 - self.transaction_cost)) / current_price
                self.current_position = units_to_buy
                self.current_capital = 0.0
                self.trades.append({
                    'time': self.historical_df.index[self._current_step],
                    'type': 'BUY',
                    'price': current_price,
                    'units': units_to_buy
                })
                logger.debug(f"[{self.historical_df.index[self._current_step]}] BUY @ {current_price:.2f}")
        elif action == 2: # SELL
            if self.current_position > 0:
                self.current_capital = self.current_position * current_price * (1 - self.transaction_cost)
                self.trades.append({
                    'time': self.historical_df.index[self._current_step],
                    'type': 'SELL',
                    'price': current_price,
                    'units': self.current_position
                })
                self.current_position = 0
                logger.debug(f"[{self.historical_df.index[self._current_step]}] SELL @ {current_price:.2f}")

        self.portfolio_value = self.current_capital
        if self.current_position > 0:
            self.portfolio_value += self.current_position * current_price

        self.equity_curve.append(self.portfolio_value)

        reward = self._calculate_reward()

        terminated = self._current_step >= len(self.historical_df) - 1
        truncated = False

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _calculate_reward(self):
        if len(self.equity_curve) < 2:
            return 0.0

        if self.reward_type == 'portfolio_return':
            return (self.equity_curve[-1] - self.equity_curve[-2]) / self.equity_curve[-2]
        elif self.reward_type == 'pnl':
            return self.equity_curve[-1] - self.equity_curve[-2]
        else:
            return 0.0

    def render(self):
        pass

    def close(self):
        pass
