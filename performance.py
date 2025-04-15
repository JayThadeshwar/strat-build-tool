import pandas as pd
import numpy as np
from typing import Dict, List
from base_strategy import TradingConfig


class PerformanceAnalyzer:
    """Analyze trading strategy performance with required and optional metrics."""

    def __init__(
        self,
        initial_cash: float,
        stake: float,
        commission: float,
        optional_metrics: List[str] = None,
    ):
        self.config = TradingConfig(
            initial_capital=initial_cash, transaction_cost_bps=commission * 10000
        )
        self.stake = stake
        self.optional_metrics = optional_metrics or []
        # Available optional metrics
        self.optional_metric_functions = {
            "sortino_ratio": self._calculate_sortino_ratio,
            "win_ratio": self._calculate_win_ratio,
            "avg_trade_return": self._calculate_avg_trade_return,
            "calmar_ratio": self._calculate_calmar_ratio,
        }
        # Validate optional metrics
        for metric in self.optional_metrics:
            if metric not in self.optional_metric_functions:
                raise ValueError(f"Unsupported optional metric: {metric}")

    def _calculate_required_metrics(
        self, portfolio: pd.DataFrame, scaled_returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate mandatory performance metrics."""
        returns = portfolio["strategy_returns"]
        portfolio_value = portfolio["portfolio_value"]

        # Total return
        total_return = portfolio_value.iloc[-1] / self.config.initial_capital - 1

        # Annual return (scaled by stake)
        annual_return = scaled_returns.mean() * 252

        # Volatility
        volatility = scaled_returns.std() * np.sqrt(252)

        # Sharpe ratio
        sharpe = (
            (annual_return - self.config.risk_free_rate) / volatility
            if volatility != 0
            else 0
        )

        # Max drawdown
        rolling_max = portfolio_value.cummax()
        drawdown = (portfolio_value - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
        }

    def _calculate_sortino_ratio(
        self, portfolio: pd.DataFrame, scaled_returns: pd.Series
    ) -> float:
        """Calculate Sortino ratio (downside risk-adjusted return)."""
        downside_returns = scaled_returns[scaled_returns < 0]
        downside_volatility = (
            downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0
        )
        annual_return = scaled_returns.mean() * 252
        return (
            (annual_return - self.config.risk_free_rate) / downside_volatility
            if downside_volatility != 0
            else 0
        )

    def _calculate_win_ratio(
        self, portfolio: pd.DataFrame, scaled_returns: pd.Series
    ) -> float:
        """Calculate percentage of positive return days."""
        positive_days = scaled_returns[scaled_returns > 0].count()
        total_days = scaled_returns.count()
        return positive_days / total_days if total_days != 0 else 0

    def _calculate_avg_trade_return(
        self, portfolio: pd.DataFrame, scaled_returns: pd.Series
    ) -> float:
        """Calculate average return per trade (position change)."""
        trades = portfolio["position"].diff().abs().gt(0)
        trade_returns = scaled_returns[trades]
        return trade_returns.mean() if not trade_returns.empty else 0

    def _calculate_calmar_ratio(
        self, portfolio: pd.DataFrame, scaled_returns: pd.Series
    ) -> float:
        """Calculate Calmar ratio (annual return / abs(max drawdown))."""
        annual_return = scaled_returns.mean() * 252
        rolling_max = portfolio["portfolio_value"].cummax()
        drawdown = (portfolio["portfolio_value"] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        return annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    def analyze(self, portfolio: pd.DataFrame) -> Dict[str, float]:
        """Calculate required and selected optional metrics."""
        if portfolio.empty:
            raise ValueError("Empty portfolio data")

        returns = portfolio["strategy_returns"]
        # Scale returns by stake
        scaled_returns = returns * (self.stake / self.config.initial_capital)

        # Always calculate required metrics
        metrics = self._calculate_required_metrics(portfolio, scaled_returns)

        # Calculate optional metrics if specified
        for metric in self.optional_metrics:
            metrics[metric] = self.optional_metric_functions[metric](
                portfolio, scaled_returns
            )

        return metrics

    def compare_stakes(
        self, portfolio: pd.DataFrame, stakes: List[float]
    ) -> pd.DataFrame:
        """Compare performance across different stake sizes."""
        results = []
        for stake in stakes:
            self.stake = stake
            metrics = self.analyze(portfolio)
            metrics["stake"] = stake
            results.append(metrics)
        return pd.DataFrame(results)
