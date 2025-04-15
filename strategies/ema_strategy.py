import pandas as pd
from typing import Optional
from base_strategy import TradingStrategy, TradingConfig


class EMAStrategy(TradingStrategy):
    """Exponential Moving Average Crossover strategy."""

    def __init__(
        self,
        short_span: int = 50,
        long_span: int = 200,
        config: Optional[TradingConfig] = None,
    ):
        super().__init__(config)
        self.short_span = short_span
        self.long_span = long_span

    def generate_signals(self) -> pd.Series:
        """Generate signals based on EMA crossover."""
        signals = pd.Series(0, index=self.price_data.index)
        short_ema = (
            self.price_data["close"].ewm(span=self.short_span, adjust=False).mean()
        )
        long_ema = (
            self.price_data["close"].ewm(span=self.long_span, adjust=False).mean()
        )

        signals[short_ema > long_ema] = 1  # Buy
        signals[short_ema < long_ema] = -1  # Sell
        return signals
