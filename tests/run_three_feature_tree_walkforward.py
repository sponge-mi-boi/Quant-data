import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(r"path\to\root")
PROJECT = ROOT / 'work' / 'PythonProject1_basicbacktester' / 'Published'
sys.path[:0] = [str(PROJECT / 'src'), str(ROOT / 'work')]

from src.data_filter import get_time_period
from src.hmm_regime import (
    build_variance_correlation_trend_features,
    build_variance_dispersion_trend_features,
)
from src.svm_regime import fit_svm_regime, predict_svm_scores
from run_logistic_regime_one_cycle import HORIZON, winner_labels
import run_ml_allocator_comparison as pipeline
import run_regime_one_cycle as strategy_source
from neutral_sleeve_cache import load_or_build

FULL_PERIOD = (0, 2060)
PROFILE = os.environ.get('TREE_REGIME_UNIVERSE', 'two_sleeves')
FEATURE_SET = os.environ.get('TREE_REGIME_FEATURES', 'variance_correlation_trend')
ACTIVE = (tuple(strategy_source.SLEEVES) if PROFILE == 'all_strategies'
          else ('momentum_trending', 'cross_asset_mv'))
NEUTRALITY = {'dollar': {'param': None}, 'beta': {'roll': 30},
              'pc': {'roll': 30, 'n': 1}}
VALIDATION_WORKERS = int(os.environ.get('TREE_VALIDATION_WORKERS', '6'))
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


_WORKER_CONTEXT = None


def initialize_worker(probabilities, sleeves, index, prices, period):
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = probabilities, sleeves, index, prices, period


def evaluate_candidate(args):
    probabilities, sleeves, index, prices, period = _WORKER_CONTEXT
    c_value, gamma, half_life, rebalance, cap = args
    weights = allocations(probabilities, half_life, rebalance, cap)
    result = pipeline.metrics(pipeline.combine(sleeves, weights, index), prices, period)
    return {'C': c_value, 'gamma': gamma,
            'smoothing_half_life': half_life,
            'rebalance_every_bars': rebalance,
            'max_sleeve_weight': cap, 'validation': result}


def main():
    strategy_source.NEUTRALITY_FILTERS = NEUTRALITY
    pipeline.FULL_PERIOD = FULL_PERIOD
    cached_data, loaded = load_or_build(FULL_PERIOD, NEUTRALITY, ACTIVE)
    all_sleeves, prices, asset_returns, all_returns = cached_data
    print('loaded cached neutralized sleeves' if loaded
          else 'built and cached neutralized sleeves', flush=True)
    sleeves = {name: all_sleeves[name] for name in ACTIVE}
    labels = winner_labels(all_returns.loc[:, ACTIVE])
    index = prices.index
    market = get_time_period(['SPY'], time_peri=FULL_PERIOD).reindex(index).pct_change()['SPY']
    if FEATURE_SET == 'variance_dispersion_trend':
        features = build_variance_dispersion_trend_features(market, asset_returns)
        feature_names = ['variance', 'dispersion', 'trend']
    elif FEATURE_SET == 'variance_correlation_trend':
        features = build_variance_correlation_trend_features(market, asset_returns)
        feature_names = ['variance', 'average_correlation', 'trend']
    else:
        raise ValueError(f'Unknown TREE_REGIME_FEATURES: {FEATURE_SET}')
    runs = []
    for number, periods in enumerate(CYCLES, 1):
        train_index = index[periods['training'][0]:periods['training'][1] - HORIZON]
        validation_index = index[slice(*periods['validation'])]
        candidates = []
        for c_value, gamma in product((.1, 1., 10.), ('scale', .1, 1.)):
            model = fit_svm_regime(features.reindex(train_index), labels.reindex(train_index),
                                   c_values=(c_value,), gamma_values=(gamma,), purge_gap=HORIZON)
            probabilities = predict_svm_scores(model, features.reindex(validation_index))
            settings = [(c_value, gamma, half_life, rebalance, cap)
                        for half_life, rebalance, cap
                        in product((2, 5, 10), (5, 10, 20), (.25, .4))]
            with ProcessPoolExecutor(
                    max_workers=VALIDATION_WORKERS, initializer=initialize_worker,
                    initargs=(probabilities, sleeves, index, prices, periods['validation'])) as executor:
                candidates.extend(executor.map(evaluate_candidate, settings))
        print(f'{PROFILE} cycle {number}/5 validation complete', flush=True)
        candidates.sort(key=lambda item: (
            passed(item['validation']), item['validation']['sharpe'],
            item['validation']['total_return']), reverse=True)
        selected = candidates[0]
        development = index[periods['training'][0]:periods['validation'][1] - HORIZON]
        frozen = fit_svm_regime(
            features.reindex(development), labels.reindex(development),
            c_values=(selected['C'],), gamma_values=(selected['gamma'],), purge_gap=HORIZON)
        held_index = index[slice(*periods['held_out'])]
        probabilities = predict_svm_scores(frozen, features.reindex(held_index))
        weights = allocations(probabilities, selected['smoothing_half_life'],
                              selected['rebalance_every_bars'], selected['max_sleeve_weight'])
        held = pipeline.metrics(pipeline.combine(sleeves, weights, index), prices, periods['held_out'])
        runs.append({'run': number, 'periods': {key: list(value) for key, value in periods.items()},
                     'candidates_tested_on_validation': len(candidates), 'selected': selected,
                     'held_out': held, 'held_out_passed': passed(held)})
    keys = ('total_return', 'sharpe', 'alpha', 'position_records')
    output = {'test': f'Five-cycle three-feature Gaussian RBF SVM: {PROFILE}',
              'features': feature_names,
              'classifier': 'gaussian_rbf_svm', 'allowed_sleeves': list(ACTIVE) + ['cash'],
              'neutrality': NEUTRALITY, 'validation_candidate_count': 162,
              'validation_workers': VALIDATION_WORKERS,
              'execution': {'execution_delay_bars': 1, 'fee_per_order': .0005,
                            'slippage_per_order': .0005, 'label_purge_bars': HORIZON},
              'runs': runs,
              'average_held_out_metrics': {
                  key: float(np.mean([run['held_out'][key] for run in runs])) for key in keys},
              'held_out_pass_count': sum(run['held_out_passed'] for run in runs),
              'scientific_status': 'Diagnostic: these historical intervals were viewed earlier.'}
    feature_tag = 'dispersion' if FEATURE_SET == 'variance_dispersion_trend' else 'correlation'
    path = ROOT / 'outputs' / f'checkpoint_three_feature_svm_{feature_tag}_{PROFILE}_neutral_walkforward_summary.json'
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
