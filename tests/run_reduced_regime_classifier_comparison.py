import json
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path (__file__).resolve().parent.parent 
PROJECT = ROOT
sys.path[:0] = [str(PROJECT / "src/quant_backtester"), str(ROOT / "artifacts")]

from src.quant_backtester import get_time_period
from src.quant_backtester.hmm_regime import build_hmm_features
from src.quant_backtester.logistic_regime import fit_logistic_regime, predict_regime_probabilities
from src.quant_backtester.svm_regime import fit_svm_regime, predict_svm_scores
from run_logistic_regime_one_cycle import HORIZON, winner_labels
import run_ml_allocator_comparison as pipeline

PROFILE = os.environ.get('REGIME_COMPARISON_PROFILE', 'reduced_three')
if PROFILE == 'cmv_cmt':
    FULL_PERIOD = (0, 2060)
    TRAIN = (0, 1540)
    VALIDATION = (1540, 1800)
    HELD_OUT = (1800, 2060)
    ACTIVE = ('cross_asset_mv', 'cross_asset_momentum_trending')
    OUTPUT_NAME = 'checkpoint_cmv_cmt_regime_classifier_comparison.json'
else:
    FULL_PERIOD = (0, 1800)
    TRAIN = (0, 1280)
    VALIDATION = (1280, 1540)
    HELD_OUT = (1540, 1800)
    ACTIVE = ('momentum_trending', 'cross_asset_mv', 'cross_asset_momentum_trending')
    OUTPUT_NAME = 'checkpoint_reduced_regime_classifier_comparison.json'


def allocations(scores, half_life, rebalance, cap):
    result = scores.drop(columns='cash').clip(upper=cap)
    result = result.ewm(halflife=half_life, adjust=False).mean()
    update = np.arange(len(result)) % rebalance == 0
    result.loc[~update] = np.nan
    return result.ffill().dropna()


def ranking_key(candidate):
    result = candidate['validation']
    sharpe = result['sharpe']
    passed = (np.isfinite(sharpe) and result['total_return'] > 0
              and sharpe > 0 and result['position_records'] >= 20)
    return passed, sharpe if np.isfinite(sharpe) else -np.inf, result['total_return']


def evaluate_candidates(model_name, model_grid, predict, sleeves, prices, features, labels, index):
    train_index = index[TRAIN[0]:TRAIN[1] - HORIZON]
    validation_index = index[VALIDATION[0]:VALIDATION[1]]
    candidates = []
    for parameters, fit_model in model_grid:
        model = fit_model(features.reindex(train_index), labels.reindex(train_index))
        scores = predict(model, features.reindex(validation_index))
        for half_life, rebalance, cap in product((2, 5, 10), (5, 10, 20), (.25, .4)):
            weights = allocations(scores, half_life, rebalance, cap)
            result = pipeline.metrics(pipeline.combine(sleeves, weights, index), prices, VALIDATION)
            candidates.append({
                **parameters, 'smoothing_half_life': half_life,
                'rebalance_every_bars': rebalance, 'max_sleeve_weight': cap,
                'validation': result,
            })
    candidates.sort(key=ranking_key, reverse=True)
    return candidates[0], len(candidates)


def main():
    pipeline.FULL_PERIOD = FULL_PERIOD
    all_sleeves, prices, asset_returns, all_sleeve_returns = pipeline.build_sleeves()
    sleeves = {name: all_sleeves[name] for name in ACTIVE}
    sleeve_returns = all_sleeve_returns.loc[:, ACTIVE]
    index = prices.index
    market = get_time_period(['SPY'], time_peri=FULL_PERIOD).reindex(index).pct_change().squeeze()
    features = build_hmm_features(market, asset_returns, roll=20, half_life=20)
    labels = winner_labels(sleeve_returns)

    svm_grid = []
    for c_value, gamma in product((.01, .1, 1., 10., 100.), ('scale', .01, .1, 1.)):
        svm_grid.append((
            {'C': c_value, 'gamma': gamma},
            lambda x, y, c=c_value, g=gamma: fit_svm_regime(
                x, y, c_values=(c,), gamma_values=(g,), purge_gap=HORIZON)))
    logistic_grid = []
    for l2 in (0., .001, .01, .05, .1, .5, 1.):
        logistic_grid.append((
            {'l2': l2},
            lambda x, y, value=l2: fit_logistic_regime(
                x, y, l2=value, learning_rate=.05, max_iterations=5000, tolerance=1e-8)))

    selected_svm, svm_count = evaluate_candidates(
        'gaussian_svm', svm_grid, predict_svm_scores,
        sleeves, prices, features, labels, index)
    selected_logistic, logistic_count = evaluate_candidates(
        'logistic', logistic_grid, predict_regime_probabilities,
        sleeves, prices, features, labels, index)

    development_index = index[TRAIN[0]:VALIDATION[1] - HORIZON]
    final_index = index[HELD_OUT[0]:HELD_OUT[1]]
    results = {}
    for name, selected in (('gaussian_svm', selected_svm), ('logistic', selected_logistic)):
        if name == 'gaussian_svm':
            model = fit_svm_regime(
                features.reindex(development_index), labels.reindex(development_index),
                c_values=(selected['C'],), gamma_values=(selected['gamma'],), purge_gap=HORIZON)
            scores = predict_svm_scores(model, features.reindex(final_index))
        else:
            model = fit_logistic_regime(
                features.reindex(development_index), labels.reindex(development_index),
                l2=selected['l2'], learning_rate=.05, max_iterations=5000, tolerance=1e-8)
            scores = predict_regime_probabilities(model, features.reindex(final_index))
        weights = allocations(
            scores, selected['smoothing_half_life'],
            selected['rebalance_every_bars'], selected['max_sleeve_weight'])
        results[name] = {
            'selected': selected,
            'validation_gate_passed': ranking_key(selected)[0],
            'held_out': pipeline.metrics(
                pipeline.combine(sleeves, weights, index), prices, HELD_OUT),
            'held_out_evaluations_this_run': 1,
        }

    output = {
        'test': f'{PROFILE} Gaussian SVM versus logistic regime classifiers',
        'allowed_sleeves': list(ACTIVE) + ['cash'],
        'removed_sleeves': [name for name in all_sleeves if name not in ACTIVE],
        'periods': {'development_training': list(TRAIN),
                    'development_validation': list(VALIDATION),
                    'new_held_out': list(HELD_OUT)},
        'candidate_counts': {'gaussian_svm': svm_count, 'logistic': logistic_count},
        'selection_rule': 'passing gate first, then validation Sharpe, then validation return',
        'execution': {'execution_delay_bars': 1, 'fee_per_order': .0005,
                      'slippage_per_order': .0005, 'label_purge_bars': HORIZON},
        'results': results,
        'scientific_status': f'Both models were specified and selected before rows {HELD_OUT[0]}:{HELD_OUT[1]} were evaluated once per frozen model.',
    }
    path = ROOT / 'artifacts' / OUTPUT_NAME
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
