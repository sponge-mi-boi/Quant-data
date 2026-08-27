import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"path\to\root")
PROJECT = ROOT / 'work' / 'PythonProject1_basicbacktester' / 'Published'
sys.path[:0] = [str(PROJECT / 'src'), str(ROOT / 'work')]

from src.data_filter import get_time_period
from src.hmm_regime import (
    build_variance_correlation_trend_features,
    variance_correlation_trend_allocations)
import run_ml_allocator_comparison as pipeline

FULL_PERIOD = (0, 2060)
PERIODS = [(760, 1020), (1020, 1280), (1280, 1540), (1540, 1800), (1800, 2060)]


def main():
    pipeline.FULL_PERIOD = FULL_PERIOD
    sleeves, prices, asset_returns, _ = pipeline.build_sleeves()
    index = prices.index
    market = get_time_period(['SPY'], time_peri=FULL_PERIOD).reindex(index).pct_change()['SPY']
    features = build_variance_correlation_trend_features(market, asset_returns)
    allocations = variance_correlation_trend_allocations(features)
    strategy_allocations = allocations.drop(columns='cash')
    target = pipeline.combine(sleeves, strategy_allocations, index)
    runs = []
    for number, period in enumerate(PERIODS, 1):
        metrics = pipeline.metrics(target, prices, period)
        runs.append({'run': number, 'period': list(period), **metrics,
                     'average_allocations': allocations.reindex(index[slice(*period)]).mean().to_dict()})
    keys = ('total_return', 'sharpe', 'alpha', 'position_records')
    output = {
        'test': 'Fixed three-feature rule regime allocator, all strategies',
        'features': ['variance', 'average_correlation', 'trend'],
        'classifier': None, 'allowed_sleeves': list(sleeves) + ['cash'],
        'rules_frozen': True,
        'execution': {'execution_delay_bars': 1, 'rebalance_every_bars': 5,
                      'smoothing_half_life': 5, 'max_sleeve_weight': .4,
                      'fee_per_order': .0005, 'slippage_per_order': .0005},
        'runs': runs,
        'average_metrics': {key: float(np.mean([run[key] for run in runs])) for key in keys},
        'pass_count': sum(run['total_return'] > 0 and np.isfinite(run['sharpe'])
                          and run['sharpe'] > 0 for run in runs),
        'scientific_status': 'Diagnostic: these historical intervals were viewed in earlier research.',
    }
    path = ROOT / 'outputs' / 'checkpoint_three_feature_rule_regime_summary.json'
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
