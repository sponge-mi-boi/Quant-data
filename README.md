# Quantitative Backtesting and Validation Framework

## Overview 
- Research-oriented backtesting and validation for the development and evaluation of portfolios and systematic trading strategies
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

  Cross-sectional strategies will be implemented in the future
## Methodology
- Walk-forward testing (train/validation/test periods)
- Functional framework for modular experimentation
- Implements regime classification 
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
`pip install -r requirements.txt`

### Entry point to execute backtester 
`python -m src.run` 

## Results
- Mean reversion of SPY stocks returns 
- Cointegration of SPY pairs, beta-neutral

Results emphasize out-of-sample performance and regime stability rather than raw metrics optimization.   
### Future goals 
  - Successful creation and implementation of PCA-based portfolios, multi-pair cointegration, regime-based models, Bayesian filtering, and optimization of weights
## Repository Structure
- `src` : Core strategy logic
- `results` : Successful portfolios' trading logs, equity curves, and related analysis tools

For a comprehensive discussion of data and methodology, see https://qgspinor.com/ 