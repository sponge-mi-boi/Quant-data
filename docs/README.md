# Repository Structure

- `results`: Top-level folder which contains the best performing strategies' results.  
    - HTML files contain full descriptions of trades, represented as dataframes, and related graphs. 
      - See https://qgspinor.com/projects/version_0_3_4 
    - JSON summary of all trading strategies and performance is given. 
    - Best results have the implementation code given in `tests/results`. Execution instructions is given below.  

# Running the Top Three Strategy Experiments

Covers the three selected approaches, ranked by their reported average held-out Sharpe:

| Rank | Regime approach | Features | Average return | Average Sharpe | Average alpha | Average max drawdown | Passed |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | Adam-optimized preset rules | Autocorrelation, correlation, volatility | 19.76% | **1.720** | 0.168 | **-4.03%** | 4/5 |
| 2 | Random forest | Autocorrelation, correlation, volatility | 23.55% | **1.361** | 0.193 | -9.69% | 5/5 |
| 3 | Neural network | Correlation, liquidity, dispersion | 20.25% | **1.178** | 0.172 | -10.63% | 5/5 |

All three approaches allocate capital among these underlying trading sleeves:

- Cross-asset mean reversion: `_get_signals_mv_cross_asset`
- Time-series momentum: `_get_signals_momentum_tr`
- Cross-asset momentum: `_get_signals_momentum_cross_asset`
- Cash

## 1. Adam preset AC/correlation/volatility — rank 1

Entry point:

```text
run_three_strategy_adam_disp_corr_liq.py
```

Configuration:

- Regime method: preset economic rules; no classifier
- Features: autocorrelation, correlation, volatility
- Allocation optimizer: Adam with deterministic SPSA validation gradients
- Adam iterations: 100
- Restarts per rebalance setting: 2
- Rebalance candidates: 5, 10, and 20 bars
- Strategies: all three sleeves
- Cash: allowed
- Neutrality: none
- Windows: fixed rolling 500-bar training, 260-bar validation, and 260-bar held-out testing
- Execution delay: 1 bar
- Fee per order: 0.05%
- Slippage per order: 0.05%

Run the complete five-period experiment in PowerShell:

```powershell
$env:ALLOCATION_OPTIMIZER = "adam"
$env:PRESET_FEATURE_SET = "ac_corr_vol"
$env:ROLLING_FIXED_WINDOWS = "true"
$env:INCLUDE_TIME_MV = "false"
$env:USE_NEUTRALITY = "false"

py run_three_strategy_adam_disp_corr_liq.py
```

This runner executes all five rolling periods in a single invocation. It does not use `NESTED_OUTER_RUN`.

Expected output:

```text
..\results\checkpoint_rolling_fixed_500_260_260_three_strategy_adam_ac_corr_vol_preset_regimes_summary.json
```

Saved average results:

- Return: 19.76%
- Sharpe: 1.720
- Alpha: 0.168
- Maximum drawdown: -4.03%
- Passed: 4/5 periods

Generated interactive report:

```text
rolling_fixed_adam_ac_corr_vol_all_five_periods.html
```

## 2. Random forest — rank 2

Entry point:

```text
run_random_forest_nested_purged_one_outer.py
```

Configuration:

- Regime model: random forest
- Features: autocorrelation, correlation, volatility
- Strategies: all three sleeves
- Cash: allowed
- Neutrality: none
- Validation: nested chronological validation
- Purge: 5 bars
- Model and allocation parameters: reselected before each outer period
- Execution delay: 1 bar
- Fee per order: 0.05%
- Slippage per order: 0.05%

Run one outer period:

```powershell
$env:NESTED_MODEL_KIND = "random_forest"
$env:NESTED_FEATURE_SET = "autocorrelation_correlation_volatility"
$env:NESTED_STRATEGY_SET = "three"
$env:NESTED_OUTER_RUN = "1"

py run_random_forest_nested_purged_one_outer.py
```

Set `NESTED_OUTER_RUN` to `1`, `2`, `3`, `4`, and `5` to execute all five periods.

Run all five periods:

```powershell
$env:NESTED_MODEL_KIND = "random_forest"
$env:NESTED_FEATURE_SET = "autocorrelation_correlation_volatility"
$env:NESTED_STRATEGY_SET = "three"

1..5 | ForEach-Object {
    $env:NESTED_OUTER_RUN = $_.ToString()
    py run_random_forest_nested_purged_one_outer.py
}
```

Expected per-period outputs:

```text
..\results\checkpoint_random_forest_ac_corr_vol_nested_purged_outer_run_<1-5>_summary.json
```

Five-period aggregate:

```text
..\results\checkpoint_random_forest_ac_corr_vol_nested_purged_five_outer_summary.json
```

Saved average results:

- Return: 23.55%
- Sharpe: 1.361
- Alpha: 0.193
- Maximum drawdown: -9.69%
- Passed: 5/5 periods

Generated interactive report:

```text
best_random_forest_all_five_periods.html
```

## 3. Neural network — rank 3

Entry point:

```text
run_random_forest_nested_purged_one_outer.py
```

The runner filename is historical. Setting `NESTED_MODEL_KIND` to `neural_network` activates the neural-network implementation.

Configuration:

- Regime model: single-hidden-layer neural network
- Features: correlation, liquidity, dispersion
- Strategies: all three sleeves
- Cash: allowed
- Neutrality: none
- Validation: nested chronological validation
- Purge: 5 bars
- Model and allocation parameters: reselected before each outer period
- Execution delay: 1 bar
- Fee per order: 0.05%
- Slippage per order: 0.05%

Run one outer period:

```powershell
$env:NESTED_MODEL_KIND = "neural_network"
$env:NESTED_FEATURE_SET = "correlation_liquidity_dispersion"
$env:NESTED_STRATEGY_SET = "three"
$env:NESTED_OUTER_RUN = "1"

py run_random_forest_nested_purged_one_outer.py
```

Set `NESTED_OUTER_RUN` from `1` through `5` to execute all periods.

Run all five periods:

```powershell
$env:NESTED_MODEL_KIND = "neural_network"
$env:NESTED_FEATURE_SET = "correlation_liquidity_dispersion"
$env:NESTED_STRATEGY_SET = "three"

1..5 | ForEach-Object {
    $env:NESTED_OUTER_RUN = $_.ToString()
    py run_random_forest_nested_purged_one_outer.py
}
```

Expected per-period outputs:

```text
..\results\checkpoint_neural_network_corr_liq_disp_nested_purged_outer_run_<1-5>_summary.json
```

Five-period aggregate:

```text
..\results\checkpoint_neural_network_corr_liq_disp_nested_purged_five_outer_summary.json
```

Saved average results:

- Return: 20.25%
- Sharpe: 1.178
- Alpha: 0.172
- Maximum drawdown: -10.63%
- Passed: 5/5 periods

## Supporting implementation files

| Responsibility | File                                                    |
|---|---------------------------------------------------------|
| Trading signal implementations | `src\strategies.py`                                     |
| Rolling-fixed preset rules and Adam/SPSA optimization | `run_three_strategy_adam_disp_corr_liq.py`              |
| Random-forest and neural-network orchestration | `run_random_forest_nested_purged_one_outer.py`          |
| Neural-network fitting and prediction helpers | `run_cmv_mt_cmt_logistic_regime.py`                     |
| Regime allocations and sleeve combination | `run_cmv_mt_cmt_logistic_regime.py`                     |
| Position normalization and net returns | `run_cross_momentum_timeseries_momentum_rule_regime.py` |
| Performance metrics, fees, and slippage | `run_cmv_full_three_stage_five_cycles.py`               |

## Methodology note

The rank-1 preset result is not directly equivalent to ranks 2 and 3:

- Rank 1 uses the rolling-fixed 500/260/260 protocol and passed 4/5 periods.
- Ranks 2 and 3 use nested purged chronological validation and passed 5/5 periods.

The ordering above is strictly by the reported average held-out Sharpe. It should not be interpreted as a perfectly controlled comparison between identical validation procedures.

## Interpretation warning

These summaries describe historical held-out windows that were viewed during earlier research. The source constituent universe also has survivorship bias. Treat the results as diagnostic research, not as an untouched estimate of future performance.
