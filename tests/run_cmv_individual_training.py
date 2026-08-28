import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path (__file__).resolve().parent.parent
PROJECT = ROOT
sys.path[:0] = [str(PROJECT / 'src'/'quant_backtester'), str(ROOT / 'tests')]

from src.quant_backtester.strategies import get_time_period
from src.quant_backtester.strategies import _get_signals_mv_cross_asset

TRAINING = (0, 500)
FEE = .0005
SLIPPAGE = .0005


def main():
    universe = pd.read_parquet(
        PROJECT / 'data' / 'close_1d_10y.parquet').columns.tolist()
    prices = get_time_period(universe, time_peri=TRAINING)
    parameters = {'z_threshold': 2.0}
    params = {'stock_list': universe, 'time_period': TRAINING, 'freq': 'd',
              'strat_class': {'cross_asset_mv': parameters}, 'parameters_': parameters}
    raw = _get_signals_mv_cross_asset(params).reindex(prices.index).fillna(0.0)
    positions = raw.shift(1).fillna(0.0)
    returns = prices.pct_change().fillna(0.0)
    turnover = positions.diff().abs().fillna(positions.abs())
    net_returns = positions * returns - turnover * (FEE + SLIPPAGE)
    total_return = ((1.0 + net_returns).prod() - 1.0) * 100.0
    sharpe = net_returns.mean().div(net_returns.std().replace(0.0, np.nan)) * np.sqrt(252)
    trades = positions.diff().fillna(positions).ne(0.0).sum()
    results = pd.DataFrame({
        'asset': universe,
        'total_return_percent': total_return.reindex(universe).to_numpy(),
        'sharpe': sharpe.reindex(universe).to_numpy(),
        'signal_changes': trades.reindex(universe).to_numpy(dtype=int),
    })
    results['eligible'] = (
        np.isfinite(results.sharpe) & (results.total_return_percent > 0)
        & (results.sharpe > 0) & (results.signal_changes >= 20))
    results = results.sort_values(
        ['eligible', 'sharpe', 'total_return_percent'], ascending=False).reset_index(drop=True)
    csv_path = ROOT / 'artifacts' / 'cmv_individual_training_results.csv'
    results.to_csv(csv_path, index=False)
    summary = {
        'test': 'Cross-sectional mean reversion on individual assets, training only',
        'training_period': list(TRAINING), 'universe_size': len(universe),
        'execution': {'execution_delay_bars': 1, 'fee_per_order': FEE,
                      'slippage_per_order': SLIPPAGE},
        'eligibility': {'positive_return': True, 'positive_finite_sharpe': True,
                        'minimum_signal_changes': 20},
        'eligible_asset_count': int(results.eligible.sum()),
        'top_10': results.head(10).to_dict(orient='records'),
        'full_results_csv': str(csv_path),
        'validation_run': False, 'held_out_run': False,
    }
    json_path = ROOT / 'artifacts' / 'cmv_individual_training_summary.json'
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
