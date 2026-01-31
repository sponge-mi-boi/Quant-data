# Quant Bot

## Overview 
This is a backtesting simulator, where various user-defined strategies can be tested on various types of data across all time scales. The results can be analyzed by way of visualization with graphs and user-defined performance metrics.
## Methodology
- Use of vectorized operations, including Pandas and NumPy
- Use of multiprocessing to speed up parameter testing
which is implemented by the 'joblib' class
- Simulated backtesting with the use of the 'vectorbt' 
- Roll-forward analysis used for validity testing
## Results
- The goal is 1.5 Alpha, 2.0 Sharpe, < 20 % drawdown, ≈ 0 beta, all of which are averaged across several roll-forward timeframes.
- Example of the resulting graphs and metrics shown https://qgspinor.com/projects/coding/version_1_%202_graphs 
  - Note that this is merely an example of a possible strategy which works along with the general framework, but is not cross-tested across regimes. 
## Repository Structure
- 'src' -Core strategy logic
- 'results' -Most relevant results
- For a full analysis of data and methodology, see https://qgspinor.com/ 