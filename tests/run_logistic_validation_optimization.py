import json
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
from run_logistic_regime_one_cycle import HORIZON, winner_labels
import run_ml_allocator_comparison as pipeline

FULL_PERIOD = (0, 1540)
TRAIN = (0, 1020)
VALIDATION = (1020, 1280)
HELD_OUT = (1280, 1540)


def allocations(probabilities, smoothing_half_life, rebalance_every, max_sleeve_weight):
    result = probabilities.drop(columns='cash').clip(upper=max_sleeve_weight)
    result = result.ewm(halflife=smoothing_half_life, adjust=False).mean()
    update = np.arange(len(result)) % rebalance_every == 0
    result.loc[~update] = np.nan
    return result.ffill().dropna()


def ranking_key(candidate):
    result = candidate['validation']
    sharpe = result['sharpe']
    eligible = (
        np.isfinite(sharpe) and result['total_return'] > 0 and sharpe > 0
        and result['position_records'] >= 20)
    return eligible, sharpe if np.isfinite(sharpe) else -np.inf, result['total_return']


def fit(features, labels, index, l2):
    return fit_logistic_regime(
        features.reindex(index), labels.reindex(index), l2=l2,
        learning_rate=.05, max_iterations=5000, tolerance=1e-8)


def main():
    pipeline.FULL_PERIOD = FULL_PERIOD
    sleeves, prices, asset_returns, sleeve_returns = pipeline.build_sleeves()
    index = prices.index
    market = get_time_period(['SPY'], time_peri=FULL_PERIOD).reindex(index).pct_change().squeeze()
    features = build_hmm_features(market, asset_returns, roll=20, half_life=20)
    labels = winner_labels(sleeve_returns)
    train_index = index[TRAIN[0]:TRAIN[1] - HORIZON]
    validation_index = index[VALIDATION[0]:VALIDATION[1]]

    candidates = []
    models = {}
    for l2 in (0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0):
        model = fit(features, labels, train_index, l2)
        models[l2] = model
        probabilities = predict_regime_probabilities(model, features.reindex(validation_index))
        for half_life, rebalance, cap in product((2, 5, 10), (5, 10, 20), (0.25, 0.4)):
            weights = allocations(probabilities, half_life, rebalance, cap)
            result = pipeline.metrics(pipeline.combine(sleeves, weights, index), prices, VALIDATION)
            candidates.append({
                'l2': l2, 'smoothing_half_life': half_life,
                'rebalance_every_bars': rebalance, 'max_sleeve_weight': cap,
                'validation': result,
            })

    candidates.sort(key=ranking_key, reverse=True)
    selected = candidates[0]
    development_index = index[TRAIN[0]:VALIDATION[1] - HORIZON]
    selected_model = fit(features, labels, development_index, selected['l2'])

    held_index = index[HELD_OUT[0]:HELD_OUT[1]]
    held_probabilities = predict_regime_probabilities(selected_model, features.reindex(held_index))
    held_weights = allocations(
        held_probabilities, selected['smoothing_half_life'],
        selected['rebalance_every_bars'], selected['max_sleeve_weight'])
    held_result = pipeline.metrics(pipeline.combine(sleeves, held_weights, index), prices, HELD_OUT)

    output = {
        'test': 'Validation-optimized multinomial logistic classifier with one fresh final test',
        'periods': {'development_training': list(TRAIN),
                    'development_validation': list(VALIDATION),
                    'new_held_out': list(HELD_OUT)},
        'search_space': {
            'l2': [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
            'smoothing_half_life': [2, 5, 10],
            'rebalance_every_bars': [5, 10, 20],
            'max_sleeve_weight': [0.25, 0.4],
            'candidate_count': len(candidates),
        },
        'selection_rule': 'passing gate first, then validation Sharpe, then validation return',
        'selected': selected,
        'refit_model': {'iterations': selected_model['iterations'], 'loss': selected_model['loss']},
        'validation_gate_passed': ranking_key(selected)[0],
        'held_out': held_result,
        'execution': {'execution_delay_bars': 1, 'fee_per_order': .0005,
                      'slippage_per_order': .0005, 'label_purge_bars': HORIZON},
        'held_out_evaluations_this_run': 1,
        'scientific_status': 'Rows 1280:1540 were not used by earlier experiments in this research session and were evaluated once after configuration freeze.',
    }
    path = ROOT / 'artifacts' / 'checkpoint_logistic_validation_optimized_summary.json'
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
