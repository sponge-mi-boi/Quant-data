import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path (__file__).resolve().parent.parent 
PROJECT = ROOT 
sys.path[:0] = [str(PROJECT / 'src'/'quant_backtester'), str(ROOT / 'tests')]

from quant_backtester import get_time_period
from quant_backtester.hmm_regime import (
    build_variance_correlation_trend_features,
    variance_correlation_trend_allocations)
import run_ml_allocator_comparison as pipeline

FULL_PERIOD = (0, 2060)
PERIODS = [(760, 1020), (1020, 1280), (1280, 1540), (1540, 1800), (1800, 2060)]
ACTIVE = ('momentum_trending', 'cross_asset_mv')


def main():
    pipeline.FULL_PERIOD = FULL_PERIOD
    all_sleeves, prices, asset_returns, _ = pipeline.build_sleeves()
    sleeves = {name: all_sleeves[name] for name in ACTIVE}
    index = prices.index
    market = get_time_period(['SPY'], time_peri=FULL_PERIOD).reindex(index).pct_change()['SPY']
    features = build_variance_correlation_trend_features(market, asset_returns)
    full_rule = variance_correlation_trend_allocations(features)
    allocations = full_rule.loc[:, list(ACTIVE) + ['cash']]
    allocations = allocations.div(allocations.sum(axis=1), axis=0)
    target = pipeline.combine(sleeves, allocations.drop(columns='cash'), index)
    runs = []
    for number, period in enumerate(PERIODS, 1):
        result = pipeline.metrics(target, prices, period)
        period_index = index[slice(*period)]
        runs.append({'run': number, 'period': list(period), **result,
                     'average_allocations': allocations.reindex(period_index).mean().to_dict()})
    keys = ('total_return', 'sharpe', 'alpha', 'position_records')
    output = {
        'test': 'Three-feature fixed rule: time-series momentum, cross-sectional MV, cash',
        'features': ['variance', 'average_correlation', 'trend'],
        'classifier': None, 'allowed_sleeves': list(ACTIVE) + ['cash'],
        'full_allocation': True,
        'execution': {'execution_delay_bars': 1, 'rebalance_every_bars': 5,
                      'smoothing_half_life': 5, 'fee_per_order': .0005,
                      'slippage_per_order': .0005},
        'runs': runs,
        'average_metrics': {key: float(np.mean([run[key] for run in runs])) for key in keys},
        'average_allocations': allocations.reindex(index[760:2060]).mean().to_dict(),
        'pass_count': sum(run['total_return'] > 0 and np.isfinite(run['sharpe'])
                          and run['sharpe'] > 0 for run in runs),
        'scientific_status': 'Diagnostic: these historical intervals were viewed in earlier research.',
    }
    path = ROOT / 'artifacts' / 'checkpoint_three_feature_momentum_cmv_cash_summary.json'
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
