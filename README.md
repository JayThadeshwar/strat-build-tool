# Trading Strategy Backtesting Package

## Overview
This Python package provides a flexible framework for backtesting trading strategies on financial data, with support for loading data from Yahoo Finance, executing trades, analyzing performance, and optimizing strategy parameters. It includes implementations of Simple Moving Average (SMA) and Exponential Moving Average (EMA) crossover strategies, a generic backtester, and a robust performance analysis module with required and optional metrics.

## Features
- **Data Loading**: Fetches historical price data from Yahoo Finance or accepts custom DataFrames.
- **Strategies**:
  - Simple Moving Average (SMA) Crossover: Generates buy/sell signals based on short and long SMA crossovers.
  - Exponential Moving Average (EMA) Crossover: Similar to SMA but uses EMAs for faster trend detection.
- **Generic Backtester**: Tests any strategy with customizable parameter grids, producing sorted results by performance metrics.
- **Performance Analysis**:
  - **Required Metrics**: Total Return, Annual Return, Sharpe Ratio, Max Drawdown, Volatility.
  - **Optional Metrics**: Sortino Ratio, Win Ratio, Average Trade Return, Calmar Ratio (configurable).
- **Extensibility**: Easily add new strategies or performance metrics without modifying core components.
- **Output**: Saves results to CSV files and provides detailed console output with best configurations and strategy comparisons.

## Package Structure
```
trading_strategy/
├── base_strategy.py    # Core trading strategy class and data loading
├── ma_strategy.py      # Simple Moving Average (SMA) strategy
├── ema_strategy.py     # Exponential Moving Average (EMA) strategy
├── performance.py      # Performance analysis with required/optional metrics
├── backtester.py       # Generic backtester for parameter optimization
├── main.py             # Example script to run SMA and EMA backtests
├── backtest_results/   # Output directory for CSV results
├── pyproject.toml      # Poetry configuration and dependency management
├── poetry.lock         # Lock file for reproducible dependencies
└── README.md           # Documentation
```

## Installation
1. **Clone the Repository** (or copy the files to a directory):
   ```bash
   git clone <repository-url>
   cd trading_strategy
   ```

2. **Install Poetry** (if not already installed):

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
   Follow the instructions to add Poetry to your PATH.

3. **Install Dependencies using Poetry**:
   ```bash
   poetry install
   ```
   
   This will:
   - Create a virtual environment (if not already active)
   - Install pandas, yfinance, and plotly with pinned versions
   - Set up the project for development

4. **Activate the Virtual Environment**
   ```bash
   eval $(poetry env activate)
   ```

5. **Verify Setup**:
   Ensure all Python files (`base_strategy.py`, `ma_strategy.py`, `ema_strategy.py`, `performance.py`, `backtester.py`, `main.py`) are in the same directory.

## Usage
The package is designed to be run via `main.py`, which demonstrates backtesting the SMA and EMA strategies on AAPL data.

### Running the Backtest
```bash
python main.py
```

This will:
- Fetch AAPL data from Yahoo Finance (2020-01-01 to 2023-12-31).
- Backtest SMA and EMA strategies with parameter combinations (short windows: 20, 50, 100; long windows: 100, 200, 300).
- Calculate required metrics (Total Return, Annual Return, Sharpe Ratio, Max Drawdown, Volatility) and optional metrics (Sortino Ratio, Win Ratio, Calmar Ratio).
- Output results to console and save them to `backtest_results/AAPL_SimpleMAStrategy_backtest.csv` and `backtest_results/AAPL_EMAStrategy_backtest.csv`.
- Display the best configuration for each strategy and a comparison table.

### Example Output
```
=== Backtesting SimpleMAStrategy ===
Results for AAPL (SimpleMAStrategy)
 short_window long_window total_return annual_return sharpe_ratio max_drawdown sortino_ratio win_ratio calmar_ratio
           20         200       0.1234        0.0345       0.2500      -0.3000        0.3500    0.4500       0.1150
           50         200       0.0875        0.0264       0.0500      -0.4210        0.0700    0.4200       0.0627
 ...

Results saved to 'backtest_results/AAPL_SimpleMAStrategy_backtest.csv'
Best SimpleMAStrategy Configuration
Short Window: 20
Long Window: 200
Total Return:   12.34%
Annual Return:  3.45%
Sharpe Ratio:   0.25
Max Drawdown:   -30.00%
Sortino Ratio: 0.3500
Win Ratio: 0.4500
Calmar Ratio: 0.1150

=== Backtesting EMAStrategy ===
Results for AAPL (EMAStrategy)
 short_span long_span total_return annual_return sharpe_ratio max_drawdown sortino_ratio win_ratio calmar_ratio
         20       200       0.1350        0.0380       0.2800      -0.2800        0.4000    0.4600       0.1357
         ...

=== Strategy Comparison ===
 Strategy  Sharpe Ratio  Total Return  Max Drawdown  Sortino Ratio  Calmar Ratio
      SMA        0.2500        0.1234       -0.3000         0.3500        0.1150
      EMA        0.2800        0.1350       -0.2800         0.4000        0.1357
```

### Customizing Parameters
Edit `main.py` to modify:
- **Ticker**: Change `ticker="AAPL"` to another stock (e.g., "MSFT", "GOOGL").
- **Date Range**: Adjust `start` and `end` (e.g., `start="2018-01-01"`, `end="2024-12-31"`).
- **Parameters**: Update `short_windows` and `long_windows` (e.g., `[10, 30, 60]`, `[150, 250, 400]`).
- **Financial Settings**: Modify `initial_cash`, `stake`, `commission`.
- **Optional Metrics**: Change `optional_metrics` (e.g., `['sortino_ratio', 'avg_trade_return']`).

## Extending the Package
### Adding a New Strategy
1. Create a new file (e.g., `rsi_strategy.py`):
   ```python
   from base_strategy import TradingStrategy, TradingConfig
   import pandas as pd
   from typing import Optional

   class RSIStrategy(TradingStrategy):
       def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30, config: Optional[TradingConfig] = None):
           super().__init__(config)
           self.period = period
           self.overbought = overbought
           self.oversold = oversold

       def generate_signals(self) -> pd.Series:
           # Implement RSI calculation and signals
           pass
   ```

2. Update `main.py` to include the new strategy:
   ```python
   from rsi_strategy import RSIStrategy
   rsi_param_grid = [{'period': p, 'overbought': ob, 'oversold': os} for p in [14, 21] for ob in [70, 80] for os in [20, 30]]
   run_strategy_backtest(backtester, RSIStrategy, "RSIStrategy", rsi_param_grid)
   ```

### Adding a New Performance Metric
1. In `performance.py`, add a new method and register it:
   ```python
   def _calculate_new_metric(self, portfolio: pd.DataFrame, scaled_returns: pd.Series) -> float:
       # Implement metric logic
       return value

   self.optional_metric_functions = {
       ...,  # Existing metrics
       'new_metric': self._calculate_new_metric
   }
   ```

2. Include it in `main.py`:
   ```python
   backtester = Backtester(..., optional_metrics=['sortino_ratio', 'new_metric'])
   ```

## Requirements
- Python 3.9+
- Libraries: `yfinance`, `pandas`, `numpy`, `plotly`
- Internet connection for Yahoo Finance data

## Notes
- **Data Source**: Relies on Yahoo Finance; ensure a stable connection.
- **Performance**: Backtests run sequentially; parallel processing could be added for large parameter grids.
- **Error Handling**: Gracefully skips invalid parameter sets or data issues, with clear error messages.

## License
This project is licensed under the MIT License.

## Contributing
Feel free to submit issues or pull requests to enhance strategies, metrics, or performance. Suggestions for new features (e.g., visualizations, additional data sources) are welcome!