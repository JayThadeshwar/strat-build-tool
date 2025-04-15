import pandas as pd
from typing import List, Dict, Type, Optional
from base_strategy import TradingStrategy, TradingConfig
from performance import PerformanceAnalyzer
import os

class Backtester:
    """Generic tool to backtest trading strategies with different parameters."""

    def __init__(
        self,
        ticker: str,
        start: str,
        end: str,
        initial_cash: float = 100000.0,
        stake: float = 50000.0,
        commission: float = 0.001,
        optional_metrics: List[str] = None,
    ):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.config = TradingConfig(
            initial_capital=initial_cash, transaction_cost_bps=commission * 10000
        )
        self.stake = stake
        self.analyzer = PerformanceAnalyzer(
            initial_cash, stake, commission, optional_metrics=optional_metrics
        )

    def run_backtest(
        self, strategy_class: Type[TradingStrategy], param_grid: List[Dict[str, any]]
    ) -> pd.DataFrame:
        """Run backtest for a strategy with given parameter combinations."""
        results = []

        for params in param_grid:
            # Set up strategy with parameters
            try:
                strategy = strategy_class(config=self.config, **params)

                # Load data and execute trades
                strategy.load_data(ticker=self.ticker, start=self.start, end=self.end)
                strategy.execute_trades()

                # Analyze performance
                metrics = self.analyzer.analyze(strategy.portfolio)
                metrics.update(params)  # Include parameters in results
                results.append(metrics)

            except Exception as e:
                print(f"Error for params {params}: {e}")
                continue

        # Convert results to DataFrame
        if not results:
            return pd.DataFrame()

        results_df = pd.DataFrame(results)
        # Ensure key metrics are always included
        columns = (
            list(params.keys())
            + ["total_return", "annual_return", "sharpe_ratio", "max_drawdown"]
            + [m for m in self.analyzer.optional_metrics if m in results_df.columns]
        )
        columns = [col for col in columns if col in results_df.columns]
        return results_df[columns].sort_values("sharpe_ratio", ascending=False)

    def save_backtest_results(
        self,
        results: pd.DataFrame,
        strategy_name: str,
        output_dir: str = "backtest_results",
    ) -> None:
        """Save backtest results to CSV."""
        if results.empty:
            return
        os.makedirs(output_dir, exist_ok=True)
        results.to_csv(
            os.path.join(output_dir, f"{self.ticker}_{strategy_name}_backtest.csv"),
            index=False,
        )
