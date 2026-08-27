import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(r"path\to\root")
PROJECT = ROOT / "work" / "PythonProject1_basicbacktester" / "Published"
sys.path[:0] = [str(PROJECT / "src"), str(ROOT / "work")]

from src import get_time_period
from src import build_hmm_features
from src import fit_svm_regime, predict_svm_scores
from run_logistic_regime_one_cycle import HORIZON, winner_labels
import run_ml_allocator_comparison as pipeline

FULL_PERIOD = (0, 1280)
TRAIN = (0, 760)
VALIDATION = (760, 1020)
HELD_OUT = (1020, 1280)


def allocations(scores, smoothing_half_life, rebalance_every, max_sleeve_weight):
    result = scores.drop(columns='cash').clip(upper=max_sleeve_weight)
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
    # Prefer candidates passing the economic gate, then Sharpe, then return.
    return eligible, sharpe if np.isfinite(sharpe) else -np.inf, result['total_return']


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
    for c_value, gamma_value in product((0.01, 0.1, 1.0, 10.0, 100.0), ('scale', 0.01, 0.1, 1.0)):
        model = fit_svm_regime(
            features.reindex(train_index), labels.reindex(train_index),
            c_values=(c_value,), gamma_values=(gamma_value,), purge_gap=HORIZON)
        models[(c_value, gamma_value)] = model
        scores = predict_svm_scores(model, features.reindex(validation_index))
        for half_life, rebalance, cap in product((2, 5, 10), (5, 10, 20), (0.25, 0.4)):
            weights = allocations(scores, half_life, rebalance, cap)
            result = pipeline.metrics(pipeline.combine(sleeves, weights, index), prices, VALIDATION)
            candidates.append({
                'C': c_value, 'gamma': gamma_value,
                'smoothing_half_life': half_life,
                'rebalance_every_bars': rebalance,
                'max_sleeve_weight': cap,
                'validation': result,
            })

    candidates.sort(key=ranking_key, reverse=True)
    selected = candidates[0]
    # Hyperparameters are now frozen. Refit using all development data, while
    # purging labels whose forward horizon would cross into the new final test.
    development_index = index[TRAIN[0]:VALIDATION[1] - HORIZON]
    selected_model = fit_svm_regime(
        features.reindex(development_index), labels.reindex(development_index),
        c_values=(selected['C'],), gamma_values=(selected['gamma'],), purge_gap=HORIZON)

    # Freeze the validation-selected configuration and evaluate held-out exactly once.
    held_index = index[HELD_OUT[0]:HELD_OUT[1]]
    held_scores = predict_svm_scores(selected_model, features.reindex(held_index))
    held_weights = allocations(
        held_scores, selected['smoothing_half_life'],
        selected['rebalance_every_bars'], selected['max_sleeve_weight'])
    held_result = pipeline.metrics(pipeline.combine(sleeves, held_weights, index), prices, HELD_OUT)

    output = {
        'test': 'Validation-optimized RBF SVM followed by one frozen held-out evaluation',
        'periods': {'development_training': list(TRAIN),
                    'development_validation': list(VALIDATION),
                    'new_held_out': list(HELD_OUT)},
        'search_space': {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'gamma': ['scale', 0.01, 0.1, 1.0],
            'smoothing_half_life': [2, 5, 10],
            'rebalance_every_bars': [5, 10, 20],
            'max_sleeve_weight': [0.25, 0.4],
            'candidate_count': len(candidates),
        },
        'selection_rule': 'passing gate first, then validation Sharpe, then validation return',
        'selected': selected,
        'selected_training_cv_balanced_accuracy': selected_model['cv_balanced_accuracy'],
        'validation_gate_passed': ranking_key(selected)[0],
        'held_out': held_result,
        'execution': {'execution_delay_bars': 1, 'fee_per_order': .0005,
                      'slippage_per_order': .0005, 'label_purge_bars': HORIZON},
        'held_out_evaluations_this_run': 1,
        'scientific_status': 'Rows 1020:1280 were not used by any earlier experiment in this research session and were evaluated once after configuration freeze.',
    }
    path = ROOT / 'outputs' / 'checkpoint_svm_validation_optimized_summary.json'
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
