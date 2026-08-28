import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("error", category=RuntimeWarning)
os.environ.setdefault('NUMBA_DISABLE_JIT', '1')

ROOT = Path (__file__).resolve().parent.parent
PROJECT = ROOT 
sys.path[:0] = [str(PROJECT / 'src'/'quant_backtester'), str(ROOT / 'tests')]

from src.quant_backtester import get_time_period

from src.quant_backtester.strategies import (
    _get_signals_momentum_tr, _get_signals_momentum_cross_asset,
    _get_signals_mv_cross_asset,
    _project_onto_constraint_nullspace)

FEE = .0005
SLIPPAGE = .0005
ROLL = 30
TOP_ASSETS = 10
PROFILE = os.environ.get('THREE_STAGE_STRATEGY', 'cross_asset_mv')
USE_NEUTRALITY = os.environ.get('APPLY_NEUTRALITY', 'true').lower() == 'true'
CYCLES = [
    {'training': (0, 500), 'validation': (500, 760), 'held_out': (760, 1020)},
    {'training': (0, 760), 'validation': (760, 1020), 'held_out': (1020, 1280)},
    {'training': (0, 1020), 'validation': (1020, 1280), 'held_out': (1280, 1540)},
    {'training': (0, 1280), 'validation': (1280, 1540), 'held_out': (1540, 1800)},
    {'training': (0, 1540), 'validation': (1540, 1800), 'held_out': (1800, 2060)},
]


def performance(net, market):
    net = pd.Series(net).fillna(0.0)
    market = pd.Series(market, index=net.index).fillna(0.0)
    wealth = (1.0 + net).cumprod()
    drawdown = wealth.div(wealth.cummax()).sub(1.0)
    std = net.std()
    sharpe = net.mean() / std * np.sqrt(252) if std > 0 else np.nan
    market_var = market.var()
    beta = net.cov(market) / market_var if market_var > 0 else 0.0
    alpha = (net.mean() - beta * market.mean()) * 252
    return {'total_return': float((wealth.iloc[-1] - 1.0) * 100.0),
            'sharpe': float(sharpe), 'alpha': float(alpha),
            'max_drawdown': float(drawdown.min() * 100.0)}


def net_returns(weights, returns):
    executed = weights.shift(1).fillna(0.0)
    turnover = executed.diff().abs().sum(axis=1).fillna(executed.abs().sum(axis=1))
    return (executed * returns[weights.columns]).sum(axis=1) - turnover * (FEE + SLIPPAGE)


def individual_ranking(raw, returns, index):
    positions = raw.reindex(index).fillna(0.0)
    executed = positions.shift(1).fillna(0.0)
    turnover = executed.diff().abs().fillna(executed.abs())
    net = executed * returns.reindex(index) - turnover * (FEE + SLIPPAGE)
    total = ((1.0 + net).prod() - 1.0) * 100.0
    sharpe = net.mean().div(net.std().replace(0.0, np.nan)) * np.sqrt(252)
    changes = executed.diff().fillna(executed).ne(0).sum()
    result = pd.DataFrame({'return': total, 'sharpe': sharpe, 'changes': changes})
    eligible = result[np.isfinite(result.sharpe) & (result['return'] > 0)
                      & (result.sharpe > 0) & (result.changes >= 20)]
    return eligible.sort_values(['sharpe', 'return'], ascending=False)


def constraint_cache(selected, raw, returns, market):
    market_variance = market.rolling(ROLL, min_periods=ROLL).var()
    beta = returns[selected].rolling(ROLL, min_periods=ROLL).cov(market).div(
        market_variance, axis=0).to_numpy()
    values = returns[selected].to_numpy()
    correlations = np.full((len(returns), len(selected), len(selected)), np.nan)

    for absolute in range(ROLL, len(returns)):
        correlations[absolute] = np.corrcoef(values[absolute - ROLL + 1:absolute + 1], rowvar=False)

    return {'assets': selected, 'locations': {asset: i for i, asset in enumerate(selected)},
            'beta': beta, 'correlations': correlations,
            'raw': raw[selected].to_numpy()}


def neutral_weights(combo, cache, index, start, stop):
    assets = list(combo)
    locations = np.array([cache['locations'][asset] for asset in assets])
    values = np.zeros((stop - start, len(assets)))
    if not USE_NEUTRALITY:
        values = cache['raw'][start:stop, locations].copy()
        gross = np.abs(values).sum(axis=1)
        active = gross > 0
        values[active] /= gross[active, None]
        return pd.DataFrame(values, index=index[start:stop], columns=assets)
    for absolute in range(max(start, ROLL), stop):
        correlation = cache['correlations'][absolute][np.ix_(locations, locations)]
        if not np.isfinite(correlation).all():
            continue
        _, vectors = np.linalg.eigh(correlation)
        constraints = np.vstack([
            np.ones(len(assets)), cache['beta'][absolute, locations], vectors[:, -1]])
        projected = _project_onto_constraint_nullspace(
            cache['raw'][absolute, locations], constraints)
        gross = np.abs(projected).sum()
        if gross > 0:
            values[absolute - start] = projected / gross
    return pd.DataFrame(values, index=index[start:stop], columns=assets)


def main():
    universe = pd.read_parquet(
        PROJECT / 'data' / 'close_1d_10y.parquet').columns.tolist()
    prices = get_time_period(universe, time_peri=(0, 2060))
    market_prices = get_time_period(['SPY'], time_peri=(0, 2060)).reindex(prices.index)['SPY']
    returns = prices.pct_change().fillna(0.0)
    market = market_prices.pct_change().fillna(0.0)
    if PROFILE == 'momentum_trending':
        strategy = {'z_threshold': 1.999, 'roll': 30}
    elif PROFILE == 'cross_asset_momentum_trending':
        strategy = {'z_threshold': 1.9283, 'roll': 35}
    else:
        strategy = {'z_threshold': 2.0}
    params = {'stock_list': universe, 'time_period': (0, 2060), 'freq': 'd',
              'strat_class': {PROFILE: strategy}, 'parameters_': strategy}
    if PROFILE == 'momentum_trending':
        raw = _get_signals_momentum_tr(params, prices)
    elif PROFILE == 'cross_asset_momentum_trending':
        raw = _get_signals_momentum_cross_asset(params)
    else:
        raw = _get_signals_mv_cross_asset(params)
    raw = raw.reindex(prices.index).fillna(0.0)
    runs = []
    for run_number, periods in enumerate(CYCLES, 1):
        train_slice = slice(*periods['training'])
        ranking = individual_ranking(raw, returns, prices.index[train_slice])
        selected = ranking.head(TOP_ASSETS).index.tolist()
        cache = constraint_cache(selected, raw, returns, market)
        validation_start, validation_stop = periods['validation']
        candidates = []
        for size in range(4, TOP_ASSETS + 1):
            for combo in combinations(selected, size):
                weights = neutral_weights(combo, cache, prices.index,
                                          validation_start - ROLL, validation_stop)
                validation_index = prices.index[validation_start:validation_stop]
                net = net_returns(weights, returns).reindex(validation_index).fillna(0.0)
                metrics = performance(net, market.reindex(validation_index))
                candidates.append((combo, metrics))
        passing = [item for item in candidates if np.isfinite(item[1]['sharpe'])
                   and item[1]['total_return'] > 0 and item[1]['sharpe'] > 0]
        pool = passing if passing else candidates
        winner, validation_metrics = max(
            pool, key=lambda item: (item[1]['sharpe'], item[1]['total_return']))
        held_start, held_stop = periods['held_out']
        weights = neutral_weights(winner, cache, prices.index,
                                  held_start - ROLL, held_stop)
        held_index = prices.index[held_start:held_stop]
        held_net = net_returns(weights, returns).reindex(held_index).fillna(0.0)
        held_metrics = performance(held_net, market.reindex(held_index))
        runs.append({'run': run_number,
                     'periods': {key: list(value) for key, value in periods.items()},
                     'universe_size': len(universe), 'training_selected_assets': selected,
                     'validation_combinations_tested': len(candidates),
                     'validation_winner': list(winner),
                     'validation': validation_metrics, 'held_out': held_metrics})
    metric_names = ('total_return', 'sharpe', 'alpha', 'max_drawdown')
    averages = {stage: {metric: float(np.mean([run[stage][metric] for run in runs]))
                        for metric in metric_names} for stage in ('validation', 'held_out')}
    output = {'test': f'Five-cycle individual training filter, combination validation, held-out {PROFILE}',
              'strategy': PROFILE,
              'classifier': None, 'neutrality': ['dollar', 'rolling_beta', 'leading_pc'],
              'selection': {'training_top_assets': TOP_ASSETS,
                            'validation_combination_sizes': [4, 10],
                            'validation_rule': 'positive return and Sharpe, then maximum Sharpe'},
              'execution': {'execution_delay_bars': 1, 'fee_per_order': FEE,
                            'slippage_per_order': SLIPPAGE},
              'runs': runs, 'average_metrics': averages,
              'pass_counts': {
                  'validation': sum(run['validation']['total_return'] > 0 and run['validation']['sharpe'] > 0 for run in runs),
                  'held_out': sum(run['held_out']['total_return'] > 0 and run['held_out']['sharpe'] > 0 for run in runs),
                  'total_runs': len(runs),
              },
              'scientific_status': 'Diagnostic: these historical windows were viewed earlier.'}
    output['neutrality'] = ['dollar', 'rolling_beta', 'leading_pc'] if USE_NEUTRALITY else []
    suffix = 'neutral' if USE_NEUTRALITY else 'nonneutral'
    path = ROOT / 'artifacts' / f'checkpoint_{PROFILE}_{suffix}_three_stage_five_cycles_summary.json'
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
