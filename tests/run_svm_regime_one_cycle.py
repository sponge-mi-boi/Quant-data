import json
import sys
from pathlib import Path

ROOT = Path(r"path\to\root")
PROJECT = ROOT / "work" / "PythonProject1_basicbacktester" / "Published"
sys.path[:0] = [str(PROJECT / "src"), str(ROOT / "work")]

from src import get_time_period
from src import build_hmm_features
from src import fit_svm_regime, predict_svm_scores
from run_logistic_regime_one_cycle import HORIZON, allocations_from_probabilities, winner_labels
from run_ml_allocator_comparison import (
    FULL_PERIOD, TRAIN, VALIDATION, HELD_OUT, build_sleeves, combine, metrics, passed)


def main():
    sleeves, prices, asset_returns, sleeve_returns = build_sleeves()
    index = prices.index
    market = get_time_period(['SPY'], time_peri=FULL_PERIOD).reindex(index).pct_change().squeeze()
    features = build_hmm_features(market, asset_returns, roll=20, half_life=20)
    labels = winner_labels(sleeve_returns)
    train_index = index[TRAIN[0]:TRAIN[1] - HORIZON]
    model = fit_svm_regime(features.reindex(train_index), labels.reindex(train_index), purge_gap=HORIZON)

    validation_index = index[VALIDATION[0]:VALIDATION[1]]
    validation_scores = predict_svm_scores(model, features.reindex(validation_index))
    validation_allocations = allocations_from_probabilities(validation_scores)
    validation_result = metrics(combine(sleeves, validation_allocations, index), prices, VALIDATION)
    gate = passed(validation_result)

    held_result = None
    held_allocations = None
    if gate:
        held_index = index[HELD_OUT[0]:HELD_OUT[1]]
        held_scores = predict_svm_scores(model, features.reindex(held_index))
        held_allocations = allocations_from_probabilities(held_scores)
        held_result = metrics(combine(sleeves, held_allocations, index), prices, HELD_OUT)

    output = {
        'test': 'RBF SVM strategy-regime classifier - one checkpoint cycle',
        'previous_regime_model_used': False,
        'features': model['feature_columns'], 'classes': model['state_columns'],
        'target': 'best positive cost-adjusted sleeve return over next 5 bars; otherwise cash',
        'model': {'kernel': 'rbf', 'class_weight': 'balanced', 'probability_calibration': False,
                  'C': model['c'], 'gamma': model['gamma'],
                  'purged_cv_balanced_accuracy': model['cv_balanced_accuracy'],
                  'purged_cv_fold_scores': model['cv_fold_scores'], 'purge_gap': model['purge_gap']},
        'execution': {'execution_delay_bars': 1, 'rebalance_every_bars': 5,
                      'smoothing_half_life': 5, 'max_sleeve_weight': .4,
                      'fee_per_order': .0005, 'slippage_per_order': .0005},
        'validation': validation_result, 'validation_passed': gate,
        'average_validation_scores': validation_scores.mean().to_dict(),
        'average_validation_allocations': validation_allocations.mean().to_dict(),
        'held_out': held_result, 'held_out_deployment': 'strategy' if gate else 'cash',
        'average_held_out_allocations': None if held_allocations is None else held_allocations.mean().to_dict(),
        'scientific_status': 'Diagnostic: this chronological held-out interval was viewed in earlier experiments.',
    }
    path = ROOT / 'outputs' / 'checkpoint_svm_regime_one_cycle_summary.json'
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
