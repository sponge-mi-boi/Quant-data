import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path (__file__).resolve().parent.parent 
PROJECT = ROOT
sys.path[:0] = [str(PROJECT / "src/quant_backtester"), str(ROOT / "artifacts")]

from src.quant_backtester import get_time_period
from src.quant_backtester.hmm_regime import build_hmm_features
from src.quant_backtester.logistic_regime import fit_logistic_regime, predict_regime_probabilities
from run_ml_allocator_comparison import (
    FULL_PERIOD, TRAIN, VALIDATION, HELD_OUT, build_sleeves, combine, metrics, passed)

HORIZON = 5


def forward_returns(returns, horizon):
    future = returns.shift(-1)
    return future.iloc[::-1].rolling(horizon, min_periods=horizon).sum().iloc[::-1]


def winner_labels(sleeve_returns):
    future = forward_returns(sleeve_returns, HORIZON)
    labels = pd.DataFrame(0.0, index=future.index, columns=list(future.columns) + ['cash'])
    valid = future.dropna().index
    winners = future.loc[valid].idxmax(axis=1)
    positive = future.loc[valid].max(axis=1) > 0
    for date in valid:
        label = winners.loc[date] if positive.loc[date] else 'cash'
        labels.loc[date, label] = 1.0
    return labels.replace(0.0, np.nan).where(labels.eq(1.0), 0.0).loc[valid]


def allocations_from_probabilities(probabilities):
    allocations = probabilities.drop(columns='cash').clip(upper=.4)
    allocations = allocations.ewm(halflife=5, adjust=False).mean()
    update = np.arange(len(allocations)) % 5 == 0
    allocations.loc[~update] = np.nan
    return allocations.ffill().dropna()


def main():
    sleeves, prices, asset_returns, sleeve_returns = build_sleeves()
    index = prices.index
    market = get_time_period(['SPY'], time_peri=FULL_PERIOD).reindex(index).pct_change().squeeze()
    features = build_hmm_features(market, asset_returns, roll=20, half_life=20)
    labels = winner_labels(sleeve_returns)
    train_index = index[TRAIN[0]:TRAIN[1] - HORIZON]
    model = fit_logistic_regime(
        features.reindex(train_index), labels.reindex(train_index),
        l2=.05, learning_rate=.05, max_iterations=5000)

    validation_index = index[VALIDATION[0]:VALIDATION[1]]
    validation_probabilities = predict_regime_probabilities(model, features.reindex(validation_index))
    validation_allocations = allocations_from_probabilities(validation_probabilities)
    validation_result = metrics(combine(sleeves, validation_allocations, index), prices, VALIDATION)
    gate = passed(validation_result)

    held_result = None
    held_allocations = None
    if gate:
        held_index = index[HELD_OUT[0]:HELD_OUT[1]]
        held_probabilities = predict_regime_probabilities(model, features.reindex(held_index))
        held_allocations = allocations_from_probabilities(held_probabilities)
        held_result = metrics(combine(sleeves, held_allocations, index), prices, HELD_OUT)

    output = {
        'test': 'Multinomial logistic strategy-regime classifier - one checkpoint cycle',
        'previous_regime_model_used': False,
        'features': model['feature_columns'],
        'classes': model['state_columns'],
        'target': 'best positive cost-adjusted sleeve return over next 5 bars; otherwise cash',
        'model': {'l2': model['l2'], 'iterations': model['iterations'], 'loss': model['loss']},
        'execution': {'execution_delay_bars': 1, 'rebalance_every_bars': 5,
                      'smoothing_half_life': 5, 'max_sleeve_weight': .4,
                      'fee_per_order': .0005, 'slippage_per_order': .0005},
        'validation': validation_result, 'validation_passed': gate,
        'average_validation_probabilities': validation_probabilities.mean().to_dict(),
        'average_validation_allocations': validation_allocations.mean().to_dict(),
        'held_out': held_result, 'held_out_deployment': 'strategy' if gate else 'cash',
        'average_held_out_allocations': None if held_allocations is None else held_allocations.mean().to_dict(),
        'scientific_status': 'Diagnostic: this chronological held-out interval was viewed in earlier experiments.',
    }
    path = ROOT / 'artifacts' / 'checkpoint_logistic_regime_one_cycle_summary.json'
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
