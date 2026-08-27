import json
import sys
from pathlib import Path

ROOT  = Path(r"path\to\root")
PROJECT = ROOT / "work" / "PythonProject1_basicbacktester" / "Published"
sys.path[:0] = [str(PROJECT / "src"), str(ROOT / "work")]

from src import get_time_period
from src import (
    build_hmm_features, filtered_state_probabilities, fit_gaussian_hmm,
    probability_weighted_allocations, state_sleeve_weights)
from run_ml_allocator_comparison import (
    FULL_PERIOD, TRAIN, VALIDATION, HELD_OUT, build_sleeves, combine, metrics, passed)


def main():
    sleeves, prices, asset_returns, sleeve_returns = build_sleeves()
    index = prices.index
    market_returns = get_time_period(['SPY'], time_peri=FULL_PERIOD).reindex(index).pct_change().squeeze()
    features = build_hmm_features(market_returns, asset_returns, roll=20, half_life=20)
    train_index = index[TRAIN[0]:TRAIN[1]]
    model = fit_gaussian_hmm(features.reindex(train_index), states=3, seed=7)
    train_probabilities = filtered_state_probabilities(model, features.reindex(train_index))
    mapping = state_sleeve_weights(
        train_probabilities, sleeve_returns.reindex(train_index), max_sleeve_weight=.4)

    validation_index = index[VALIDATION[0]:VALIDATION[1]]
    validation_initial = train_probabilities.iloc[-1].to_numpy() @ model['transition']
    validation_probabilities = filtered_state_probabilities(
        model, features.reindex(validation_index), validation_initial)
    validation_allocations = probability_weighted_allocations(
        validation_probabilities, mapping, rebalance_every=5, smoothing_half_life=5)
    validation_target = combine(sleeves, validation_allocations, index)
    validation_result = metrics(validation_target, prices, VALIDATION)
    gate = passed(validation_result)

    held_result = None
    held_allocations = None
    if gate:
        held_index = index[HELD_OUT[0]:HELD_OUT[1]]
        held_initial = validation_probabilities.iloc[-1].to_numpy() @ model['transition']
        held_probabilities = filtered_state_probabilities(model, features.reindex(held_index), held_initial)
        held_allocations = probability_weighted_allocations(
            held_probabilities, mapping, rebalance_every=5, smoothing_half_life=5)
        held_target = combine(sleeves, held_allocations, index)
        held_result = metrics(held_target, prices, HELD_OUT)

    state_feature_means = model['state_means'] * model['scale'] + model['mean']
    output = {
        'test': 'Three-state Gaussian HMM regime allocator - one checkpoint cycle',
        'features': model['feature_columns'],
        'model': {
            'states': 3, 'iterations': model['iterations'],
            'transition_matrix': model['transition'].tolist(),
            'state_feature_means': state_feature_means.tolist(),
            'state_sleeve_mapping': mapping.to_dict(orient='index'),
        },
        'execution': {
            'execution_delay_bars': 1, 'rebalance_every_bars': 5,
            'smoothing_half_life': 5, 'max_sleeve_weight': .4,
            'fee_per_order': .0005, 'slippage_per_order': .0005,
        },
        'validation': validation_result,
        'validation_passed': gate,
        'average_validation_allocations': validation_allocations.mean().to_dict(),
        'held_out': held_result,
        'held_out_deployment': 'strategy' if gate else 'cash',
        'average_held_out_allocations': None if held_allocations is None else held_allocations.mean().to_dict(),
        'scientific_status': 'Diagnostic: this chronological held-out interval was viewed in earlier experiments.',
    }
    path = ROOT / 'outputs' / 'checkpoint_hmm_regime_one_cycle_summary.json'
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
