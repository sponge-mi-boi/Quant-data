# Quantitative Backtesting and Validation Framework

## Overview

- Research-oriented backtesting and validation for the development and evaluation of portfolios and systematic trading
  strategies
- Currently focused on U.S.-based equities and derivatives (primarily SPY-related assets)

### Supported Asset Classes

- Stocks/ETFs
- Options/Futures

### Supported Time Horizons

- Medium frequency (daily/hourly)
- Low frequency (1mo/3mo)

### Supported Strategy Classes

- Time-series strategies
    - Mean reversion
    - Cointegration (statistical arbitrage)
    - Momentum/ trend following
- Cross-sectional strategies
    - Mean reversion
    - Momentum/ trend following

## Methodology

- Walk-forward testing (train/validation/test periods)
- Functional framework for modular experimentation
- Implements regime classification
- Utilizes common portfolio constraints 
- Vectorized operations (Pandas/NumPy)
- Multiprocessing (joblib)
- Bayesian optimization (optuna)
- Risk-adjusted performance metrics used for assessment
- Supports graphs for visualization along with HTML export (Plotly)

## Quickstart

### Create the virtual environment

`python -m venv .venv`

### Activate the virtual environment

Windows (PowerShell)

`.venv\Scripts\Activate.ps1`

macOS / Linux (bash)

`source .venv/bin/activate`

### Install necessary modules

`python -m pip install -e `

### Entry point to execute backtester

`python -m src.run`

## Results

- Regime based strategy implementation of mean reversion, momentum trending, cross asset mean reversion, cross asset momentum trending
- Multi asset cointegration with neutrality condition(s)

Results emphasize out-of-sample performance and regime stability rather than raw metrics optimization.

### Future goals

- Implementation of a broader range of factor models
- More explicit Bayesian filtering with a detailed implementation of the Kalman filter and path dependent models
- Greater focus on volatility including the implementation of volatility targeting and classification of asset volatility 

## Repository Structure

- `src` : Core strategy logic
- `docs` : Successful portfolios' trading logs, equity curves, and related analysis tools
    - `results`: Human-readable results.
- `examples`: Example(s) of a successful backtesting process.
- `data`: cache data storage
- `artifacts`: cache results storage

For a comprehensive discussion of data and methodology, see https://qgspinor.com/ 