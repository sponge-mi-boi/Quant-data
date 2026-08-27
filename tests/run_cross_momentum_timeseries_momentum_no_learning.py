"""Fixed 50/50 cross-sectional and time-series momentum walk-forward test."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault('NUMBA_DISABLE_JIT', '1')
ROOT = Path(r'path\to\root')
PROJECT = ROOT / 'work' / 'PythonProject1_basicbacktester' / 'Published'
sys.path[:0] = [str(PROJECT / 'src'), str(ROOT / 'work')]

from src.data_filter import get_time_period
from src.strategies import (
    _get_signals_momentum_cross_asset, _get_signals_momentum_tr)
from run_cmv_full_three_stage_five_cycles import CYCLES, FEE, SLIPPAGE, performance


def normalize(raw, assets):
    weights = raw[assets].copy()
    return weights.div(weights.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def net_returns(target, returns):
    executed = target.shift(1).fillna(0.0)
    turnover = executed.diff().abs().sum(axis=1).fillna(executed.abs().sum(axis=1))
    return (executed * returns[target.columns]).sum(axis=1) - turnover * (FEE + SLIPPAGE)


def passed(metrics):
    return bool(np.isfinite(metrics['sharpe']) and
                metrics['total_return'] > 0 and metrics['sharpe'] > 0)


def main():
    cross_source = json.loads((ROOT / 'outputs' /
        'checkpoint_cross_asset_momentum_trending_nonneutral_three_stage_five_cycles_summary.json').read_text())
    time_source = json.loads((ROOT / 'outputs' /
        'checkpoint_momentum_trending_nonneutral_three_stage_five_cycles_summary.json').read_text())

    universe = pd.read_parquet(PROJECT / 'data' / 'processed' / 'close_1d_10y.parquet').columns.tolist()
    prices = get_time_period(universe, time_peri=(0, 2060))
    returns = prices.pct_change().fillna(0.0)
    market = get_time_period(['SPY'], time_peri=(0, 2060)).reindex(prices.index)['SPY'].pct_change().fillna(0.0)

    cross_params = {'stock_list': universe, 'time_period': (0, 2060), 'freq': 'd',
        'strat_class': {'cross_asset_momentum_trending': {'z_threshold': 1.9283, 'roll': 35}},
        'parameters_': {'z_threshold': 1.9283, 'roll': 35}}
    time_params = {'stock_list': universe, 'time_period': (0, 2060), 'freq': 'd',
        'strat_class': {'momentum_trending': {'z_threshold': 1.999, 'roll': 30}},
        'parameters_': {'z_threshold': 1.999, 'roll': 30}}
    cross_raw = _get_signals_momentum_cross_asset(cross_params).reindex(prices.index).fillna(0.0)
    time_raw = _get_signals_momentum_tr(time_params, prices).reindex(prices.index).fillna(0.0)

    runs = []
    for number, (periods, cross_run, time_run) in enumerate(
            zip(CYCLES, cross_source['runs'], time_source['runs']), 1):
        cross = normalize(cross_raw, cross_run['validation_winner'])
        time = normalize(time_raw, time_run['validation_winner'])
        columns = list(dict.fromkeys([*cross.columns, *time.columns]))
        target = pd.DataFrame(0.0, index=prices.index, columns=columns)
        target.loc[:, cross.columns] += 0.5 * cross
        target.loc[:, time.columns] += 0.5 * time
        held = prices.index[slice(*periods['held_out'])]
        metrics = performance(net_returns(target, returns).reindex(held).fillna(0.0),
                              market.reindex(held))
        runs.append({'run': number,
                     'cross_momentum_assets': cross_run['validation_winner'],
                     'time_series_momentum_assets': time_run['validation_winner'],
                     'held_out': metrics, 'held_out_passed': passed(metrics)})

    names = ('total_return', 'sharpe', 'alpha', 'max_drawdown')
    output = {
        'test': 'Fixed 50/50 cross-sectional momentum + time-series momentum, no learning',
        'classifier': None, 'regime_filter': None, 'neutrality': [],
        'allocation': {'cross_asset_momentum_trending': 0.5, 'momentum_trending': 0.5},
        'execution': {'execution_delay_bars': 1, 'fee_per_order': FEE,
                      'slippage_per_order': SLIPPAGE},
        'runs': runs,
        'average_held_out_metrics': {
            name: float(np.mean([run['held_out'][name] for run in runs])) for name in names},
        'held_out_pass_count': sum(run['held_out_passed'] for run in runs),
        'scientific_status': 'Diagnostic: these historical windows were viewed earlier.'}
    path = ROOT / 'outputs' / 'checkpoint_cross_momentum_timeseries_momentum_no_learning_summary.json'
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
