import json
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(r"path\to\root")
PROJECT = ROOT / 'work' / 'PythonProject1_basicbacktester' / 'Published'
sys.path[:0] = [str(PROJECT / 'src'), str(ROOT / 'work')]

from src import get_time_period
from src import build_variance_correlation_trend_features
from src import fit_logistic_regime, predict_regime_probabilities
from run_logistic_regime_one_cycle import HORIZON, winner_labels
import run_ml_allocator_comparison as pipeline
import run_regime_one_cycle as strategy_source

FULL_PERIOD = (0, 2060)
PROFILE = os.environ.get('LOGISTIC_REGIME_UNIVERSE', 'two_sleeves')
ACTIVE = (tuple(strategy_source.SLEEVES) if PROFILE == 'all_strategies'
          else ('momentum_trending', 'cross_asset_mv'))
NEUTRALITY = {'dollar': {'param': None}, 'beta': {'roll': 30},
              'pc': {'roll': 30, 'n': 1}}
CYCLES = [
    {'training': (0, 500), 'validation': (500, 760), 'held_out': (760, 1020)},
    {'training': (0, 760), 'validation': (760, 1020), 'held_out': (1020, 1280)},
    {'training': (0, 1020), 'validation': (1020, 1280), 'held_out': (1280, 1540)},
    {'training': (0, 1280), 'validation': (1280, 1540), 'held_out': (1540, 1800)},
    {'training': (0, 1540), 'validation': (1540, 1800), 'held_out': (1800, 2060)},
]


def allocations(probabilities, half_life, rebalance, cap):
    result = probabilities.drop(columns='cash').clip(upper=cap)
    result = result.ewm(halflife=half_life, adjust=False).mean()
    update = np.arange(len(result)) % rebalance == 0
    result.loc[~update] = np.nan
    return result.ffill().dropna()


def passed(metrics):
    return bool(np.isfinite(metrics['sharpe']) and metrics['total_return'] > 0
                and metrics['sharpe'] > 0 and metrics['position_records'] >= 20)


def fit(features, labels, index, l2):
    return fit_logistic_regime(
        features.reindex(index), labels.reindex(index), l2=l2,
        learning_rate=.05, max_iterations=5000, tolerance=1e-8)


def main():
    strategy_source.NEUTRALITY_FILTERS = NEUTRALITY
    pipeline.FULL_PERIOD = FULL_PERIOD
    all_sleeves, prices, asset_returns, all_returns = pipeline.build_sleeves()
    sleeves = {name: all_sleeves[name] for name in ACTIVE}
    sleeve_returns = all_returns.loc[:, ACTIVE]
    index = prices.index
    market = get_time_period(['SPY'], time_peri=FULL_PERIOD).reindex(index).pct_change()['SPY']
    features = build_variance_correlation_trend_features(market, asset_returns)
    labels = winner_labels(sleeve_returns)
    runs = []
    for number, periods in enumerate(CYCLES, 1):
        train_index = index[periods['training'][0]:periods['training'][1] - HORIZON]
        validation_index = index[slice(*periods['validation'])]
        candidates = []
        for l2 in (0., .001, .01, .05, .1, .5, 1.):
            model = fit(features, labels, train_index, l2)
            probabilities = predict_regime_probabilities(model, features.reindex(validation_index))
            for half_life, rebalance, cap in product((2, 5, 10), (5, 10, 20), (.25, .4)):
                weights = allocations(probabilities, half_life, rebalance, cap)
                result = pipeline.metrics(pipeline.combine(sleeves, weights, index), prices,
                                          periods['validation'])
                candidates.append({'l2': l2, 'smoothing_half_life': half_life,
                                   'rebalance_every_bars': rebalance,
                                   'max_sleeve_weight': cap, 'validation': result})
        candidates.sort(key=lambda item: (
            passed(item['validation']),
            item['validation']['sharpe'] if np.isfinite(item['validation']['sharpe']) else -np.inf,
            item['validation']['total_return']), reverse=True)
        selected = candidates[0]
        development_index = index[periods['training'][0]:periods['validation'][1] - HORIZON]
        frozen_model = fit(features, labels, development_index, selected['l2'])
        held_index = index[slice(*periods['held_out'])]
        held_probabilities = predict_regime_probabilities(
            frozen_model, features.reindex(held_index))
        held_weights = allocations(
            held_probabilities, selected['smoothing_half_life'],
            selected['rebalance_every_bars'], selected['max_sleeve_weight'])
        held = pipeline.metrics(pipeline.combine(sleeves, held_weights, index),
                                prices, periods['held_out'])
        runs.append({'run': number,
                     'periods': {key: list(value) for key, value in periods.items()},
                     'candidates_tested_on_validation': len(candidates),
                     'selected': selected, 'held_out': held,
                     'held_out_passed': passed(held),
                     'average_held_out_allocations': held_weights.mean().to_dict()})
    keys = ('total_return', 'sharpe', 'alpha', 'position_records')
    output = {
        'test': f'Five-cycle three-feature logistic regime allocator: {PROFILE}',
        'features': ['variance', 'average_correlation', 'trend'],
        'classifier': 'multinomial_logistic_regression',
        'allowed_sleeves': list(ACTIVE) + ['cash'], 'neutrality': NEUTRALITY,
        'target': 'best positive cost-adjusted sleeve return over next 5 bars; otherwise cash',
        'validation_search': {'candidate_count': 126,
                              'selection': 'passing gate, then Sharpe, then return'},
        'execution': {'execution_delay_bars': 1, 'fee_per_order': .0005,
                      'slippage_per_order': .0005, 'label_purge_bars': HORIZON},
        'runs': runs,
        'average_held_out_metrics': {
            key: float(np.mean([run['held_out'][key] for run in runs])) for key in keys},
        'held_out_pass_count': sum(run['held_out_passed'] for run in runs),
        'scientific_status': 'Diagnostic: these historical intervals were viewed earlier.',
    }
    path = ROOT / 'outputs' / f'checkpoint_three_feature_logistic_{PROFILE}_neutral_walkforward_summary.json'
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
