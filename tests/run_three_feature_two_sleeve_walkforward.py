import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path (__file__).resolve().parent.parent 
PROJECT = ROOT 
sys.path[:0] = [str(PROJECT / 'src'/'quant_backtester'), str(ROOT / 'tests')]

from src.quant_backtester import get_time_period
from src.quant_backtester.hmm_regime import (
    build_variance_correlation_trend_features,
    variance_correlation_trend_allocations)
import run_ml_allocator_comparison as pipeline
import run_regime_one_cycle as strategy_source

FULL_PERIOD = (0, 2060)
ACTIVE = ('momentum_trending', 'cross_asset_mv')
NEUTRALITY = {'dollar': {'param': None}, 'beta': {'roll': 30},
              'pc': {'roll': 30, 'n': 1}}
CYCLES = [
    {'training': (0, 500), 'validation': (500, 760), 'held_out': (760, 1020)},
    {'training': (0, 760), 'validation': (760, 1020), 'held_out': (1020, 1280)},
    {'training': (0, 1020), 'validation': (1020, 1280), 'held_out': (1280, 1540)},
    {'training': (0, 1280), 'validation': (1280, 1540), 'held_out': (1540, 1800)},
    {'training': (0, 1540), 'validation': (1540, 1800), 'held_out': (1800, 2060)},
]


def scaled_allocations(features, half_life, rebalance, momentum_scale, cash_scale):
    weights = variance_correlation_trend_allocations(
        features, allocation_half_life=half_life, rebalance_every=rebalance)
    result = weights.loc[:, list(ACTIVE) + ['cash']].copy()
    result['momentum_trending'] *= momentum_scale
    result['cash'] *= cash_scale
    return result.div(result.sum(axis=1), axis=0)


def passed(metrics):
    return bool(np.isfinite(metrics['sharpe']) and metrics['total_return'] > 0
                and metrics['sharpe'] > 0 and metrics['position_records'] >= 20)


def main():
    strategy_source.NEUTRALITY_FILTERS = NEUTRALITY
    pipeline.FULL_PERIOD = FULL_PERIOD
    all_sleeves, prices, asset_returns, _ = pipeline.build_sleeves()
    sleeves = {name: all_sleeves[name] for name in ACTIVE}
    index = prices.index
    market = get_time_period(['SPY'], time_peri=FULL_PERIOD).reindex(index).pct_change()['SPY']
    features = build_variance_correlation_trend_features(market, asset_returns)
    runs = []
    for number, periods in enumerate(CYCLES, 1):
        candidates = []
        for half_life, rebalance, momentum_scale, cash_scale in product(
                (2, 5, 10), (5, 10, 20), (.5, 1., 2.), (.5, 1., 2.)):
            allocations = scaled_allocations(
                features, half_life, rebalance, momentum_scale, cash_scale)
            target = pipeline.combine(sleeves, allocations.drop(columns='cash'), index)
            metrics = pipeline.metrics(target, prices, periods['validation'])
            candidates.append({
                'smoothing_half_life': half_life, 'rebalance_every_bars': rebalance,
                'momentum_scale': momentum_scale, 'cash_scale': cash_scale,
                'validation': metrics,
            })
        candidates.sort(key=lambda item: (
            passed(item['validation']),
            item['validation']['sharpe'] if np.isfinite(item['validation']['sharpe']) else -np.inf,
            item['validation']['total_return']), reverse=True)
        selected = candidates[0]
        frozen = scaled_allocations(
            features, selected['smoothing_half_life'], selected['rebalance_every_bars'],
            selected['momentum_scale'], selected['cash_scale'])
        frozen_target = pipeline.combine(sleeves, frozen.drop(columns='cash'), index)
        held = pipeline.metrics(frozen_target, prices, periods['held_out'])
        held_index = index[slice(*periods['held_out'])]
        runs.append({
            'run': number, 'periods': {key: list(value) for key, value in periods.items()},
            'candidates_tested_on_validation': len(candidates), 'selected': selected,
            'held_out': held, 'held_out_passed': passed(held),
            'average_held_out_allocations': frozen.reindex(held_index).mean().to_dict(),
        })
    keys = ('total_return', 'sharpe', 'alpha', 'position_records')
    output = {
        'test': 'Five-cycle validation-selected three-feature rule allocator',
        'features': ['variance', 'average_correlation', 'trend'],
        'classifier': None, 'allowed_sleeves': list(ACTIVE) + ['cash'],
        'neutrality': NEUTRALITY,
        'validation_search': {'candidate_count': 81,
                              'selection': 'passing gate, then Sharpe, then return'},
        'execution': {'execution_delay_bars': 1, 'fee_per_order': .0005,
                      'slippage_per_order': .0005},
        'runs': runs,
        'average_held_out_metrics': {
            key: float(np.mean([run['held_out'][key] for run in runs])) for key in keys},
        'held_out_pass_count': sum(run['held_out_passed'] for run in runs),
        'scientific_status': 'Diagnostic: these historical intervals were viewed earlier.',
    }
    path = ROOT / 'artifacts' / 'checkpoint_three_feature_momentum_cmv_cash_walkforward_summary.json'
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
