"""Five outer tests: historical mega caps, time-series momentum, no regime."""
import json
import os
import sys
from itertools import product
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"path\to\root")
PROJECT = ROOT / 'work' / 'PythonProject1_basicbacktester' / 'Published'
sys.path[:0] = [str(PROJECT / 'src'), str(ROOT / 'work')]

from src import get_time_period
from src import _get_signals_momentum_tr, _get_signals_mv_cross_asset
from run_cmv_full_three_stage_five_cycles import FEE, SLIPPAGE, performance
from run_cmv_full_three_stage_five_cycles import individual_ranking
from run_cross_momentum_timeseries_momentum_rule_regime import normalize, net_returns, passed
from run_cmv_mt_cmt_bayesian_hmm_regime import select_strategy_on_validation


BASE_FOLDS = [
    {'training': (0, 280), 'validation': (280, 400)},
    {'training': (120, 400), 'validation': (400, 520)},
    {'training': (240, 520), 'validation': (520, 640)},
]
BASE_HELD = (760, 1020)
CAP_BUCKET = os.environ.get('MOMENTUM_CAP_BUCKET', 'mega').lower()
PURE_STRATEGY = os.environ.get('PURE_STRATEGY', 'momentum_trending').lower()
BEST_ASSET_ONLY = os.environ.get('PURE_BEST_ASSET_ONLY', 'false').lower() == 'true'
BASKET_SIZE = int(os.environ.get('PURE_BASKET_SIZE', '1' if BEST_ASSET_ONLY else '0'))
TRAIN_RANKING = os.environ.get('PURE_TRAIN_RANKING', 'eligible').lower()
if BASKET_SIZE < 0 or BASKET_SIZE > 10:
    raise ValueError('PURE_BASKET_SIZE must be between 0 and 10')
if TRAIN_RANKING not in {'eligible', 'sharpe_only', 'volatility_only'}:
    raise ValueError('PURE_TRAIN_RANKING must be eligible, sharpe_only, or volatility_only')
if CAP_BUCKET not in {'large', 'mega'}:
    raise ValueError('MOMENTUM_CAP_BUCKET must be large or mega')
if PURE_STRATEGY not in {'momentum_trending', 'cross_asset_mv'}:
    raise ValueError('PURE_STRATEGY must be momentum_trending or cross_asset_mv')


def safe_metric(metric):
    return {key: (float(value) if np.isfinite(value) else 0.0)
            for key, value in metric.items()}


def training_singleton_sharpe_ranking(raw, returns, index):
    """Rank singletons only by causal net training Sharpe.

    A 20-change activity floor is retained to avoid ranking nearly inactive
    assets from a handful of observations. Positive return/Sharpe hurdles are
    deliberately not imposed in this mode.
    """
    positions = raw.reindex(index).fillna(0.0)
    executed = positions.shift(1).fillna(0.0)
    turnover = executed.diff().abs().fillna(executed.abs())
    net = executed * returns.reindex(index) - turnover * (FEE + SLIPPAGE)
    sharpe = net.mean().div(net.std().replace(0.0, np.nan)) * np.sqrt(252)
    changes = executed.diff().fillna(executed).ne(0).sum()
    ranking = pd.DataFrame({'sharpe': sharpe, 'changes': changes})
    ranking = ranking[np.isfinite(ranking.sharpe) & (ranking.changes >= 20)]
    return ranking.sort_values('sharpe', ascending=False)


def training_singleton_volatility_ranking(raw, returns, index):
    """Rank sufficiently active singleton strategies by lowest net volatility."""
    positions = raw.reindex(index).fillna(0.0)
    executed = positions.shift(1).fillna(0.0)
    turnover = executed.diff().abs().fillna(executed.abs())
    net = executed * returns.reindex(index) - turnover * (FEE + SLIPPAGE)
    volatility = net.std() * np.sqrt(252)
    changes = executed.diff().fillna(executed).ne(0).sum()
    ranking = pd.DataFrame({'volatility': volatility, 'changes': changes})
    ranking = ranking[np.isfinite(ranking.volatility) & (ranking.changes >= 20)]
    return ranking.sort_values('volatility', ascending=True)


def select_exact_size_on_validation(name, candidates, returns, market, index, periods):
    train_index = index[slice(*periods['training'])]
    val_index = index[slice(*periods['validation'])]
    validation_start, validation_stop = periods['validation']
    calculation_index = index[max(0, validation_start - 2):validation_stop]
    choices = []
    for parameters, raw in candidates:
        if TRAIN_RANKING == 'sharpe_only':
            ranked = training_singleton_sharpe_ranking(raw, returns, train_index)
        elif TRAIN_RANKING == 'volatility_only':
            ranked = training_singleton_volatility_ranking(raw, returns, train_index)
        else:
            ranked = individual_ranking(raw, returns, train_index)
        top = ranked.head(10).index.tolist()
        if len(top) < BASKET_SIZE:
            continue
        for basket in combinations(top, BASKET_SIZE):
            assets = list(basket)
            frame = normalize(raw.reindex(calculation_index), assets)
            metric = safe_metric(performance(
                net_returns(frame, returns).reindex(val_index).fillna(0.0),
                market.reindex(val_index)))
            choices.append({'parameters': parameters, 'assets': assets,
                            'validation': metric})
    if not choices:
        raise ValueError(f'no validation candidates for {name}')
    choices.sort(key=lambda choice: (
        passed(choice['validation']), choice['validation']['sharpe'],
        choice['validation']['total_return']), reverse=True)
    return choices[0], len(choices)


def run_outer(outer_run, snapshots, all_prices, market):
    offset = 260 * (outer_run - 1)
    folds = [{key: (value[0] + offset, value[1] + offset)
              for key, value in fold.items()} for fold in BASE_FOLDS]
    held_period = (BASE_HELD[0] + offset, BASE_HELD[1] + offset)
    bucket_column = 'is_large' if CAP_BUCKET == 'large' else 'is_mega'
    chosen = snapshots[(snapshots.outer_run == outer_run) & snapshots[bucket_column]]
    universe = [asset for asset in all_prices.columns
                if asset in set(chosen.asset) and asset != 'SPY']
    prices = all_prices[universe]
    returns = prices.pct_change().fillna(0.0)
    grid = ([{'z_threshold': z, 'roll': roll}
             for roll, z in product((20, 30, 60), (1.5, 2.0, 2.5))]
            if PURE_STRATEGY == 'momentum_trending' else
            [{'z_threshold': z} for z in (1.5, 2.0, 2.5)])
    candidates = []
    for parameters in grid:
        config = {'stock_list': universe, 'time_period': (0, 2060), 'freq': 'd',
                  'strat_class': {PURE_STRATEGY: parameters},
                  'parameters_': parameters}
        raw = (_get_signals_momentum_tr(config, prices)
               if PURE_STRATEGY == 'momentum_trending' else
               _get_signals_mv_cross_asset(config))
        raw = raw.reindex(prices.index).fillna(0.0)
        candidates.append((parameters, raw))

    selections = []
    for fold in folds:
        try:
            selector = (select_exact_size_on_validation if BASKET_SIZE > 0
                        else select_strategy_on_validation)
            selection, candidate_count = selector(
                PURE_STRATEGY, candidates, returns, market, prices.index, fold)
            selections.append({**selection, 'candidate_count': candidate_count})
        except ValueError as exc:
            if 'no validation candidates' not in str(exc):
                raise
            selections.append({'parameters': None, 'assets': [], 'validation': None,
                               'candidate_count': 0,
                               'status': 'unavailable: no basket passed original eligibility rules'})

    final = selections[-1]
    held = prices.index[slice(*held_period)]
    if final['assets']:
        raw = next(frame for parameters, frame in candidates
                   if parameters == final['parameters'])
        target = normalize(raw, final['assets'])
    else:
        target = pd.DataFrame(0.0, index=prices.index, columns=universe)
    metric = safe_metric(performance(
        net_returns(target, returns).reindex(held).fillna(0.0), market.reindex(held)))
    result = {
        'test': f'Historical {CAP_BUCKET}-cap {PURE_STRATEGY} without regime filter',
        'outer_run': outer_run,
        'strategy': PURE_STRATEGY,
        'basket_size': BASKET_SIZE if BASKET_SIZE > 0 else '4-10',
        'training_asset_ranking': TRAIN_RANKING,
        'regime_filter': None,
        'neutrality': [],
        'cash_allowed': True,
        'universe_snapshot': {
            'method': ('point-in-time USD 10 billion <= market cap < USD 200 billion'
                       if CAP_BUCKET == 'large' else
                       'point-in-time market cap >= USD 200 billion'),
            'information_cutoff': str(chosen.information_cutoff.iloc[0]),
            'training_start': str(chosen.training_start.iloc[0]),
            'asset_count': len(universe), 'assets': universe,
            'benchmark_excluded': ['SPY'],
        },
        'inner_folds': folds,
        'fold_selections': selections,
        'final_selection': final,
        'outer_held_out': list(held_period),
        'held_out': metric,
        'held_out_passed': passed(metric),
        'execution': {'execution_delay_bars': 1, 'fee_per_order': FEE,
                      'slippage_per_order': SLIPPAGE},
        'scientific_status': ('Diagnostic: held-out intervals were viewed earlier; '
                              'the May 2026 source universe retains survivorship bias.'),
    }
    strategy_slug = ('time_series_momentum' if PURE_STRATEGY == 'momentum_trending'
                     else 'cross_sectional_mean_reversion')
    if BASKET_SIZE == 1:
        strategy_slug += '_best_asset'
    elif BASKET_SIZE > 1:
        strategy_slug += f'_{BASKET_SIZE}_assets'
    if TRAIN_RANKING == 'sharpe_only':
        strategy_slug += '_training_sharpe_top10'
    elif TRAIN_RANKING == 'volatility_only':
        strategy_slug += '_training_low_vol_top10'
    path = ROOT / 'outputs' / f'checkpoint_{strategy_slug}_historical_{CAP_BUCKET}_no_regime_outer_run_{outer_run}.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(f'outer run {outer_run}: {json.dumps(metric)}', flush=True)
    return result


def main():
    snapshots = pd.read_parquet(
        ROOT / 'work' / 'historical_market_cap' / 'pretraining_market_caps.parquet')
    full_universe = pd.read_parquet(
        PROJECT / 'data' / 'processed' / 'close_1d_10y.parquet').columns.tolist()
    all_prices = get_time_period(full_universe, time_peri=(0, 2060))
    market = get_time_period(['SPY'], time_peri=(0, 2060)).reindex(all_prices.index)['SPY'].pct_change().fillna(0.0)
    results = [run_outer(run, snapshots, all_prices, market) for run in range(1, 6)]
    metrics = ('total_return', 'sharpe', 'alpha', 'max_drawdown')
    summary = {
        'setup': f'Historical {CAP_BUCKET}-cap {PURE_STRATEGY} only{f", exact {BASKET_SIZE}-asset basket" if BASKET_SIZE > 0 else ""}; no regime and no neutrality',
        'validation_method': ('three chronological 280-bar training / 120-bar validation folds; '
                              f'training ranks singleton assets by {TRAIN_RANKING}; top 10 retained; '
                              f'{f"exactly {BASKET_SIZE} assets" if BASKET_SIZE > 0 else "4-10 asset baskets"} evaluated in validation; '
                              'the most recent third-fold selection is frozen for held-out testing; '
                              'earlier folds are stability diagnostics'),
        'runs': [{'run': result['outer_run'], 'asset_count': result['universe_snapshot']['asset_count'],
                  **result['held_out'], 'passed': result['held_out_passed']}
                 for result in results],
        'averages': {metric: float(np.mean([result['held_out'][metric] for result in results]))
                     for metric in metrics},
        'passed_runs': sum(result['held_out_passed'] for result in results),
        'total_runs': 5,
        'execution': results[0]['execution'],
        'scientific_status': results[0]['scientific_status'],
    }
    strategy_slug = ('time_series_momentum' if PURE_STRATEGY == 'momentum_trending'
                     else 'cross_sectional_mean_reversion')
    if BASKET_SIZE == 1:
        strategy_slug += '_best_asset'
    elif BASKET_SIZE > 1:
        strategy_slug += f'_{BASKET_SIZE}_assets'
    if TRAIN_RANKING == 'sharpe_only':
        strategy_slug += '_training_sharpe_top10'
    elif TRAIN_RANKING == 'volatility_only':
        strategy_slug += '_training_low_vol_top10'
    path = ROOT / 'outputs' / f'checkpoint_{strategy_slug}_historical_{CAP_BUCKET}_no_regime_five_outer_summary.json'
    path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
