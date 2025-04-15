import pandas as pd
from typing import Optional
from base_strategy import TradingStrategy, TradingConfig


class SimpleMAStrategy(TradingStrategy):
    """Moving Average Crossover strategy."""

    def __init__(
        self,
        short_window: int = 50,
        long_window: int = 200,
        config: Optional[TradingConfig] = None,
    ):
        super().__init__(config)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self) -> pd.Series:
        """Generate signals based on MA crossover."""
        signals = pd.Series(0, index=self.price_data.index)
        short_ma = self.price_data["close"].rolling(self.short_window).mean()
        long_ma = self.price_data["close"].rolling(self.long_window).mean()

        signals[short_ma > long_ma] = 1  # Buy
        signals[short_ma < long_ma] = -1  # Sell
        return signals
