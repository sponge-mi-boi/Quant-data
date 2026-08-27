import json
import sys
from itertools import product
from pathlib import Path

ROOT  = Path(r"path\to\root")
PROJECT = ROOT / 'work' / 'PythonProject1_basicbacktester' / 'Published'
sys.path[:0] = [str(PROJECT / 'src'), str(ROOT / 'work')]

from src.data_filter import get_time_period
from src.hmm_regime import build_hmm_features
from src.logistic_regime import fit_logistic_regime, predict_regime_probabilities
from src.svm_regime import fit_svm_regime, predict_svm_scores
from run_logistic_regime_one_cycle import HORIZON, winner_labels
import run_ml_allocator_comparison as pipeline
import run_reduced_regime_classifier_comparison as comparison
import run_regime_one_cycle as strategy_source

FULL_PERIOD = (0, 2320)
TRAIN = (0, 1800)
VALIDATION = (1800, 2060)
HELD_OUT = (2060, 2320)
NEUTRALITY = {'dollar': {'param': None}, 'beta': {'roll': 30}, 'pc': {'roll': 30, 'n': 1}}
UNIVERSES = {
    'all_strategies': tuple(strategy_source.SLEEVES),
    'cmv_cmt_only': ('cross_asset_mv', 'cross_asset_momentum_trending'),
}


def model_grids():
    svm = []
    for c_value, gamma in product((.01, .1, 1., 10., 100.), ('scale', .01, .1, 1.)):
        svm.append((
            {'C': c_value, 'gamma': gamma},
            lambda x, y, c=c_value, g=gamma: fit_svm_regime(
                x, y, c_values=(c,), gamma_values=(g,), purge_gap=HORIZON)))
    logistic = []
    for l2 in (0., .001, .01, .05, .1, .5, 1.):
        logistic.append((
            {'l2': l2},
            lambda x, y, value=l2: fit_logistic_regime(
                x, y, l2=value, learning_rate=.05, max_iterations=5000, tolerance=1e-8)))
    return svm, logistic


def final_evaluation(kind, selected, features, labels, sleeves, prices, index):
    development = index[TRAIN[0]:VALIDATION[1] - HORIZON]
    final_index = index[HELD_OUT[0]:HELD_OUT[1]]
    if kind == 'gaussian_svm':
        model = fit_svm_regime(
            features.reindex(development), labels.reindex(development),
            c_values=(selected['C'],), gamma_values=(selected['gamma'],), purge_gap=HORIZON)
        scores = predict_svm_scores(model, features.reindex(final_index))
    else:
        model = fit_logistic_regime(
            features.reindex(development), labels.reindex(development),
            l2=selected['l2'], learning_rate=.05, max_iterations=5000, tolerance=1e-8)
        scores = predict_regime_probabilities(model, features.reindex(final_index))
    weights = comparison.allocations(
        scores, selected['smoothing_half_life'],
        selected['rebalance_every_bars'], selected['max_sleeve_weight'])
    return pipeline.metrics(pipeline.combine(sleeves, weights, index), prices, HELD_OUT)


def main():
    strategy_source.NEUTRALITY_FILTERS = NEUTRALITY
    pipeline.FULL_PERIOD = FULL_PERIOD
    comparison.TRAIN, comparison.VALIDATION, comparison.HELD_OUT = TRAIN, VALIDATION, HELD_OUT
    all_sleeves, prices, asset_returns, all_returns = pipeline.build_sleeves()
    index = prices.index
    market = get_time_period(['SPY'], time_peri=FULL_PERIOD).reindex(index).pct_change().squeeze()
    features = build_hmm_features(market, asset_returns, roll=20, half_life=20)
    svm_grid, logistic_grid = model_grids()
    results = {}
    for universe_name, active in UNIVERSES.items():
        sleeves = {name: all_sleeves[name] for name in active}
        labels = winner_labels(all_returns.loc[:, active])
        universe_results = {}
        for kind, grid, predictor in (
            ('gaussian_svm', svm_grid, predict_svm_scores),
            ('logistic', logistic_grid, predict_regime_probabilities),
        ):
            selected, count = comparison.evaluate_candidates(
                kind, grid, predictor, sleeves, prices, features, labels, index)
            universe_results[kind] = {
                'selected': selected,
                'validation_gate_passed': comparison.ranking_key(selected)[0],
                'held_out': final_evaluation(
                    kind, selected, features, labels, sleeves, prices, index),
                'candidate_count': count,
                'held_out_evaluations_this_run': 1,
            }
        results[universe_name] = {'allowed_sleeves': list(active) + ['cash'],
                                  'models': universe_results}
    output = {
        'test': 'Simultaneously dollar-, beta-, and leading-PC-neutral regime classifiers',
        'neutrality': NEUTRALITY,
        'periods': {'development_training': list(TRAIN),
                    'development_validation': list(VALIDATION),
                    'new_held_out': list(HELD_OUT)},
        'execution': {'execution_delay_bars': 1, 'fee_per_order': .0005,
                      'slippage_per_order': .0005, 'label_purge_bars': HORIZON},
        'results': results,
        'scientific_status': 'All four models were selected and frozen before the common rows 2060:2320 test interval was evaluated once per model.',
    }
    path = ROOT / 'outputs' / 'checkpoint_neutral_regime_universe_comparison.json'
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
