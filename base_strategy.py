import pandas as pd
import numpy as np
import os
from dataclasses import dataclass
from typing import Optional, Dict
import yfinance as yf


@dataclass
class TradingConfig:
    """Configuration for trading parameters."""

    initial_capital: float = 100000.0  # Starting capital
    transaction_cost_bps: float = 0.1  # Cost per trade in basis points
    risk_free_rate: float = 0.02  # Annual risk-free rate for Sharpe ratio


class TradingStrategy:
    """Base class for trading strategies."""

    def __init__(self, config: Optional[TradingConfig] = None):
        self.config = config if config else TradingConfig()
        self.portfolio = pd.DataFrame()
        self.metrics = {}

    def load_data(
        self,
        price_data: pd.DataFrame = None,
        ticker: str = None,
        start: str = None,
        end: str = None,
    ) -> None:
        """Load price data from DataFrame or Yahoo Finance."""
        if price_data is not None:
            # Use provided DataFrame
            if not {"date", "close"}.issubset(price_data.columns):
                raise ValueError("Data must have 'date' and 'close' columns")
            self.price_data = price_data.set_index("date").sort_index()
        elif ticker and start and end:
            # Validate ticker
            if not isinstance(ticker, str) or not ticker.strip():
                raise ValueError("Ticker must be a non-empty string")

            # Fetch data from Yahoo Finance
            try:
                # Ensure ticker is uppercase and stripped
                ticker = ticker.strip().upper()
                df = yf.download(ticker, start=start, end=end, progress=False)
                if df.empty:
                    raise ValueError(f"No data retrieved for ticker {ticker}")

                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Prepare data in required format
                self.price_data = pd.DataFrame({"date": df.index, "close": df["Close"]})
                self.price_data["date"] = pd.to_datetime(self.price_data["date"])
                self.price_data = self.price_data.set_index("date").sort_index()

                # Verify data integrity
                if self.price_data["close"].isna().all():
                    raise ValueError(f"Invalid price data for ticker {ticker}")
            except Exception as e:
                raise ValueError(f"Failed to fetch data from Yahoo Finance: {str(e)}")
        else:
            raise ValueError(
                "Must provide either price_data or ticker with start and end dates"
            )

    def generate_signals(self) -> pd.Series:
        """Generate trading signals (1: buy, -1: sell, 0: hold)."""
        raise NotImplementedError("Must implement generate_signals")

    def execute_trades(self) -> None:
        """Run trades based on signals with transaction costs."""
        if not hasattr(self, "price_data"):
            raise ValueError("No price data loaded")

        # Get signals and calculate daily returns
        signals = self.generate_signals()
        returns = self.price_data["close"].pct_change().fillna(0)

        # Build portfolio
        self.portfolio = pd.DataFrame(index=self.price_data.index)
        self.portfolio["signal"] = signals
        self.portfolio["market_returns"] = returns
        self.portfolio["position"] = signals.shift(1)  # Next-day execution

        # Apply transaction costs
        trades = self.portfolio["position"].diff().abs().fillna(0)
        costs = trades * (self.config.transaction_cost_bps / 10000)
        self.portfolio["strategy_returns"] = (
            self.portfolio["position"] * returns - costs
        )

        # Calculate portfolio value
        self.portfolio["portfolio_value"] = (
            1 + self.portfolio["strategy_returns"]
        ).cumprod() * self.config.initial_capital

    def calculate_metrics(self) -> Dict:
        """Calculate key performance metrics."""
        if self.portfolio.empty:
            raise ValueError("No portfolio data available")

        returns = self.portfolio["strategy_returns"]
        portfolio_value = self.portfolio["portfolio_value"]

        # Cumulative return
        cum_return = portfolio_value.iloc[-1] / self.config.initial_capital - 1

        # Annualized metrics
        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe = (
            (annual_return - self.config.risk_free_rate) / annual_volatility
            if annual_volatility != 0
            else 0
        )

        # Maximum drawdown
        rolling_max = portfolio_value.cummax()
        drawdown = (portfolio_value - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        self.metrics = {
            "cumulative_return": cum_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "volatility": annual_volatility,
        }
        return self.metrics

    def save_results(self, output_dir: str = "results") -> None:
        """Save portfolio and metrics to CSV."""
        if self.portfolio.empty:
            return
        os.makedirs(output_dir, exist_ok=True)
        self.portfolio.to_csv(os.path.join(output_dir, "portfolio.csv"))
        pd.DataFrame([self.metrics]).to_csv(
            os.path.join(output_dir, "metrics.csv"), index=False
        )
