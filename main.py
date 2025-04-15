from backtester import Backtester
from strategies.ma_strategy import SimpleMAStrategy
from strategies.ema_strategy import EMAStrategy
import itertools
import pandas as pd


def run_strategy_backtest(backtester, strategy_class, strategy_name, param_grid):
    """Helper function to run and display backtest results for a strategy."""
    print(f"\n=== Backtesting {strategy_name} ===")
    try:
        results = backtester.run_backtest(strategy_class, param_grid)
        if results.empty:
            print(f"No valid results generated for {strategy_name}.")
            return None
    except Exception as e:
        print(f"Backtest failed for {strategy_name}: {e}")
        return None

    # Display results
    print(f"\nResults for {backtester.ticker} ({strategy_name})")
    print(results.round(4).to_string(index=False))

    # Save results
    backtester.save_backtest_results(results, strategy_name)
    print(
        f"\nResults saved to 'backtest_results/{backtester.ticker}_{strategy_name}_backtest.csv'"
    )

    # Highlight best strategy
    best = results.iloc[0]
    print(f"\nBest {strategy_name} Configuration")
    for param in [
        k for k in best.index if k not in results.columns[2:]
    ]:  # Exclude metrics
        print(f"{param.replace('_', ' ').title()}: {int(best[param])}")
    print(f"Total Return:   {best['total_return']:.2%}")
    print(f"Annual Return:  {best['annual_return']:.2%}")
    print(f"Sharpe Ratio:   {best['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:   {best['max_drawdown']:.2%}")
    for metric in backtester.analyzer.optional_metrics:
        print(f"{metric.replace('_', ' ').title()}: {best[metric]:.4f}")

    return results


def main():
    # Set up backtester with optional metrics
    backtester = Backtester(
        ticker="AAPL",
        start="2020-01-01",
        end="2023-12-31",
        initial_cash=100000.0,
        stake=50000.0,
        commission=0.001,
        optional_metrics=["sortino_ratio", "win_ratio", "calmar_ratio"],
    )

    # Define parameter grid for strategies
    short_windows = [20, 50, 100]
    long_windows = [100, 200, 300]
    param_grid = [
        {"short_window": short, "long_window": long}
        for short, long in itertools.product(short_windows, long_windows)
        if short < long
    ]
    ema_param_grid = [
        {"short_span": short, "long_span": long}
        for short, long in itertools.product(short_windows, long_windows)
        if short < long
    ]

    # Run backtests for both strategies
    sma_results = run_strategy_backtest(
        backtester, SimpleMAStrategy, "SimpleMAStrategy", param_grid
    )
    ema_results = run_strategy_backtest(
        backtester, EMAStrategy, "EMAStrategy", ema_param_grid
    )

    # Compare best results
    if sma_results is not None and ema_results is not None:
        print("\n=== Strategy Comparison ===")
        best_sma = sma_results.iloc[0]
        best_ema = ema_results.iloc[0]
        comparison = pd.DataFrame(
            {
                "Strategy": ["SMA", "EMA"],
                "Sharpe Ratio": [best_sma["sharpe_ratio"], best_ema["sharpe_ratio"]],
                "Total Return": [best_sma["total_return"], best_ema["total_return"]],
                "Max Drawdown": [best_sma["max_drawdown"], best_ema["max_drawdown"]],
                "Sortino Ratio": [
                    best_sma.get("sortino_ratio", 0),
                    best_ema.get("sortino_ratio", 0),
                ],
                "Calmar Ratio": [
                    best_sma.get("calmar_ratio", 0),
                    best_ema.get("calmar_ratio", 0),
                ],
            }
        )
        print(comparison.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
