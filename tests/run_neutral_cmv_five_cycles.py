import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT  = Path(r"path\to\root")
PROJECT = ROOT / 'work' / 'PythonProject1_basicbacktester' / 'Published'
sys.path[:0] = [str(PROJECT / 'src'), str(ROOT / 'work')]

from src.data_filter import get_time_period
from src.strategies import _get_signals_mv_cross_asset, _weights_alloc
import run_ml_allocator_comparison as pipeline

FULL_PERIOD = (0, 2060)
TOP_ASSETS = 10
FEE = .0005
SLIPPAGE = .0005
NEUTRALITY = {'dollar': {'param': None}, 'beta': {'roll': 30}, 'pc': {'roll': 30, 'n': 1}}
CYCLES = [
    {'training': (0, 500), 'validation': (500, 760), 'held_out': (760, 1020)},
    {'training': (0, 760), 'validation': (760, 1020), 'held_out': (1020, 1280)},
    {'training': (0, 1020), 'validation': (1020, 1280), 'held_out': (1280, 1540)},
    {'training': (0, 1280), 'validation': (1280, 1540), 'held_out': (1540, 1800)},
    {'training': (0, 1540), 'validation': (1540, 1800), 'held_out': (1800, 2060)},
]


def params(assets, period):
    strategy = {'z_threshold': 2.0}
    return {'stock_list': list(assets), 'time_period': period, 'freq': 'd',
            'strat_class': {'cross_asset_mv': strategy}, 'parameters_': strategy,
            'weights_filter': NEUTRALITY}


def rank_training_assets(raw, prices, index):
    returns = prices.reindex(index).pct_change().fillna(0.0)
    positions = raw.reindex(index).fillna(0.0).shift(1).fillna(0.0)
    turnover = positions.diff().abs().fillna(positions.abs())
    net = positions * returns - turnover * (FEE + SLIPPAGE)
    total = (1.0 + net).prod() - 1.0
    sharpe = net.mean().div(net.std().replace(0.0, np.nan)) * np.sqrt(252)
    trades = positions.diff().fillna(positions).ne(0.0).sum()
    ranking = pd.DataFrame({'total_return': total, 'sharpe': sharpe, 'trades': trades})
    eligible = ranking[np.isfinite(ranking.sharpe) & (ranking.total_return > 0)
                       & (ranking.sharpe > 0) & (ranking.trades >= 20)]
    if len(eligible) < TOP_ASSETS:
        eligible = ranking[np.isfinite(ranking.sharpe) & (ranking.trades >= 20)]
    return eligible.sort_values(['sharpe', 'total_return'], ascending=False)


def average_metrics(runs, stage):
    keys = ('total_return', 'sharpe', 'alpha', 'position_records')
    return {key: float(np.mean([run[stage][key] for run in runs])) for key in keys}


def main():
    universe = pd.read_parquet(
        PROJECT / 'data' / 'processed' / 'close_1d_10y.parquet').columns.tolist()
    prices = get_time_period(universe, time_peri=FULL_PERIOD)
    index = prices.index
    raw = _get_signals_mv_cross_asset(params(universe, FULL_PERIOD)).reindex(index).fillna(0.0)
    runs = []
    for number, periods in enumerate(CYCLES, 1):
        training_index = index[periods['training'][0]:periods['training'][1]]
        ranking = rank_training_assets(raw, prices, training_index)
        selected = ranking.head(TOP_ASSETS).index.tolist()
        if len(selected) != TOP_ASSETS:
            raise RuntimeError(f'run {number}: only {len(selected)} eligible assets')
        end = periods['held_out'][1]
        weights = _weights_alloc(
            params(selected, (0, end)), raw[selected].iloc[:end], NEUTRALITY)
        target = weights.reindex(index).fillna(0.0).shift(1).fillna(0.0)
        result = {'run': number,
                  'periods': {key: list(value) for key, value in periods.items()},
                  'universe_size': len(universe), 'selected_assets': selected,
                  'training_selection_metrics': ranking.loc[selected].to_dict(orient='index')}
        for stage, period in periods.items():
            result[stage] = pipeline.metrics(target, prices[selected], period)
        runs.append(result)
    output = {
        'test': 'Five expanding cycles: full-universe filter then neutral CMV',
        'strategy': 'cross_asset_mv', 'classifier': None,
        'asset_selection': {'top_assets': TOP_ASSETS,
                            'rule': 'positive training net return and Sharpe, >=20 trades; rank Sharpe then return'},
        'neutrality': NEUTRALITY,
        'execution': {'execution_delay_bars': 1, 'fee_per_order': FEE, 'slippage_per_order': SLIPPAGE},
        'runs': runs,
        'average_metrics': {stage: average_metrics(runs, stage)
                            for stage in ('training', 'validation', 'held_out')},
        'scientific_status': 'Diagnostic: historical intervals were viewed earlier; selection is training-only within each cycle.',
    }
    path = ROOT / 'outputs' / 'checkpoint_neutral_cmv_five_cycles_full_filter_summary.json'
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
