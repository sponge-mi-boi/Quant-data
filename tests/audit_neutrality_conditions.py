import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"path\to\root")

PROJECT = ROOT / "work" / "PythonProject1_basicbacktester" / "Published"
sys.path[:0] = [str(PROJECT / "src"), str(ROOT / "work")]

from src.market_filters_analysis import _beta_filter_weights, _pc_filter_weights
from src.strategies import (
    _get_signals_momentum_cross_asset, _get_signals_mv_cross_asset, _weights_alloc)
from run_regime_one_cycle import SLEEVES

PERIOD = (0, 400)
TOLERANCE = 1e-8


def residual_summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        'observations': int(len(values)),
        'mean_absolute_residual': float(np.mean(np.abs(values))) if len(values) else None,
        'maximum_absolute_residual': float(np.max(np.abs(values))) if len(values) else None,
        'passes_tolerance': bool(len(values) and np.max(np.abs(values)) <= TOLERANCE),
    }


def audit_strategy(name):
    config = SLEEVES[name]
    assets = config['assets']
    base = {
        'stock_list': assets,
        'time_period': PERIOD,
        'freq': 'd',
        'strat_class': {name: config['parameters']},
        'parameters_': config['parameters'],
    }
    raw = (_get_signals_mv_cross_asset(base)[assets] if name == 'cross_asset_mv'
           else _get_signals_momentum_cross_asset(base)[assets])
    baseline = _weights_alloc(dict(base), raw, {})
    result = {'actual_experiment_dollar_residual': residual_summary(baseline.sum(axis=1))}

    for constraint, specification in (
        ('dollar', {'dollar': {'param': None}}),
        ('beta', {'beta': {'roll': 30}}),
        ('pc', {'pc': {'roll': 30, 'n': 1}}),
    ):
        params = dict(base)
        params['weights_filter'] = specification
        try:
            constrained = _weights_alloc(params, raw, specification)
            if constraint == 'dollar':
                residual = constrained.sum(axis=1)
            elif constraint == 'beta':
                beta = _beta_filter_weights(params).reindex(constrained.index)
                common = constrained.index.intersection(beta.dropna().index)
                residual = (constrained.loc[common] * beta.loc[common]).sum(axis=1)
            else:
                pc = _pc_filter_weights(params)
                aligned = constrained.iloc[-len(pc):].to_numpy()
                residual = np.einsum('tka,ta->tk', pc, aligned).ravel()
            result[f'{constraint}_constraint'] = residual_summary(residual)
        except Exception as error:
            result[f'{constraint}_constraint'] = {
                'passes_tolerance': False,
                'error': f'{type(error).__name__}: {error}',
            }
    combined_specification = {
        'dollar': {'param': None}, 'beta': {'roll': 30}, 'pc': {'roll': 30, 'n': 1}}
    params = dict(base)
    params['weights_filter'] = combined_specification
    try:
        constrained = _weights_alloc(params, raw, combined_specification)
        beta = _beta_filter_weights(params).reindex(constrained.index)
        pc = _pc_filter_weights(params)
        common = constrained.index.intersection(beta.dropna().index)
        pc_weights = constrained.iloc[-len(pc):].to_numpy()
        result['combined_constraints'] = {
            'dollar': residual_summary(constrained.sum(axis=1)),
            'beta': residual_summary((constrained.loc[common] * beta.loc[common]).sum(axis=1)),
            'pc': residual_summary(np.einsum('tka,ta->tk', pc, pc_weights).ravel()),
        }
    except Exception as error:
        result['combined_constraints'] = {
            'passes_tolerance': False, 'error': f'{type(error).__name__}: {error}'}
    return result


def main():
    output = {
        'test': 'Direct neutrality-condition audit',
        'period': list(PERIOD),
        'tolerance': TOLERANCE,
        'strategies': {
            name: audit_strategy(name)
            for name in ('cross_asset_mv', 'cross_asset_momentum_trending')
        },
    }
    path = ROOT / 'outputs' / 'neutrality_condition_audit.json'
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
