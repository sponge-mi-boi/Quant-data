import json
import os
os.environ.setdefault('NUMBA_DISABLE_JIT', '1')
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path (__file__).resolve().parent.parent 
PROJECT = ROOT
sys.path[:0] = [str(PROJECT / "src/quant_backtester"), str(ROOT / "artifacts")]

from src.quant_backtester import get_time_period
from src.quant_backtester.hmm_regime import build_variance_dispersion_trend_features
from src.quant_backtester.strategies import _get_signals_mv_cross_asset
from src.quant_backtester.svm_regime import fit_svm_regime, predict_svm_scores
from src.quant_backtester.logistic_regime import fit_logistic_regime, predict_regime_probabilities
from src.quant_backtester.elastic_logistic_regime import (
    fit_elastic_logistic_regime, predict_elastic_probabilities)
from src.quant_backtester.decision_tree_regime import (
    fit_decision_tree_regime, predict_decision_tree_probabilities)
from run_cmv_full_three_stage_five_cycles import CYCLES, FEE, SLIPPAGE, performance

HORIZON = 5
MODEL_KIND = os.environ.get('CMV_REGIME_MODEL', 'rbf_svm')


def targets(raw, assets, exposure, returns):
    weights = raw[assets].copy()
    gross = weights.abs().sum(axis=1).replace(0, np.nan)
    weights = weights.div(gross, axis=0).fillna(0.0)
    return weights.mul(exposure.reindex(weights.index).ffill().fillna(0.0), axis=0)


def net_returns(weights, returns):
    executed = weights.shift(1).fillna(0.0)
    turnover = executed.diff().abs().sum(axis=1).fillna(executed.abs().sum(axis=1))
    return (executed * returns[weights.columns]).sum(axis=1) - turnover * (FEE + SLIPPAGE)


def exposure_from_scores(scores, threshold, half_life, rebalance):
    exposure = (scores['cross_asset_mv'] >= threshold).astype(float)
    exposure = exposure.ewm(halflife=half_life, adjust=False).mean()
    update = np.arange(len(exposure)) % rebalance == 0
    return exposure.where(update).ffill().fillna(0.0)


def passed(metrics):
    return bool(np.isfinite(metrics['sharpe']) and metrics['total_return'] > 0
                and metrics['sharpe'] > 0)


def main():
    source = json.loads((ROOT / 'artifacts' /
        'checkpoint_cross_asset_mv_nonneutral_three_stage_five_cycles_summary.json').read_text())
    universe = pd.read_parquet(PROJECT / 'data' /
                               'close_1d_10y.parquet').columns.tolist()
    prices = get_time_period(universe, time_peri=(0, 2060))
    returns = prices.pct_change().fillna(0.0)
    market = get_time_period(['SPY'], time_peri=(0, 2060)).reindex(prices.index)['SPY'].pct_change().fillna(0.0)
    params = {'stock_list': universe, 'time_period': (0, 2060), 'freq': 'd',
              'strat_class': {'cross_asset_mv': {'z_threshold': 2.0}},
              'parameters_': {'z_threshold': 2.0}}
    raw = _get_signals_mv_cross_asset(params).reindex(prices.index).fillna(0.0)
    features = build_variance_dispersion_trend_features(market, returns)
    runs = []
    for number, (periods, saved) in enumerate(zip(CYCLES, source['runs']), 1):
        assets = saved['validation_winner']
        base = targets(raw, assets, pd.Series(1.0, index=prices.index), returns)
        base_net = net_returns(base, returns)
        forward = (1 + base_net).rolling(HORIZON).apply(np.prod, raw=True).sub(1).shift(-HORIZON)
        labels = pd.DataFrame({'cross_asset_mv': (forward > 0).astype(float),
                               'cash': (forward <= 0).astype(float)}, index=prices.index)
        train = prices.index[periods['training'][0]:periods['training'][1] - HORIZON]
        val = prices.index[slice(*periods['validation'])]
        candidates = []
        model_grid = ([(c, gamma) for c, gamma in product((.1, 1., 10.), ('scale', .1, 1.))]
                      if MODEL_KIND == 'rbf_svm' else
                      ([(d, leaf) for d, leaf in product((2, 3, 4, 6), (10, 30, 60))]
                       if MODEL_KIND == 'decision_tree' else
                      ([(p, r) for p, r in product((.01, .1, 1.), (.25, .5, .75))]
                       if MODEL_KIND == 'elastic' else
                       ([(p, 1.) for p in (.001, .01, .1, 1., 10.)]
                        if MODEL_KIND == 'l1' else [(l2, None) for l2 in (.1, .3, 1., 3., 10.)]))))
        for first, second in model_grid:
            if MODEL_KIND == 'rbf_svm':
                model = fit_svm_regime(features.reindex(train), labels.reindex(train),
                                       c_values=(first,), gamma_values=(second,), purge_gap=HORIZON)
                scores = predict_svm_scores(model, features.reindex(val))
            elif MODEL_KIND == 'decision_tree':
                model = fit_decision_tree_regime(features.reindex(train), labels.reindex(train),
                                                  max_depth=first, min_samples_leaf=second)
                scores = predict_decision_tree_probabilities(model, features.reindex(val))
            elif MODEL_KIND in ('elastic', 'l1'):
                model = fit_elastic_logistic_regime(features.reindex(train), labels.reindex(train),
                                                     penalty=first, l1_ratio=second)
                scores = predict_elastic_probabilities(model, features.reindex(val))
            else:
                model = fit_logistic_regime(features.reindex(train), labels.reindex(train), l2=first)
                scores = predict_regime_probabilities(model, features.reindex(val))
            for threshold, half_life, rebalance in product((.4, .5, .6), (2, 5, 10), (5, 10, 20)):
                exposure = exposure_from_scores(scores, threshold, half_life, rebalance)
                net = net_returns(targets(raw, assets, exposure, returns), returns).reindex(val).fillna(0)
                metrics = performance(net, market.reindex(val))
                if not np.isfinite(metrics['sharpe']) and np.allclose(net, 0.0):
                    metrics['sharpe'] = 0.0
                model_params = ({'C': first, 'gamma': second} if MODEL_KIND == 'rbf_svm' else
                                ({'max_depth': first, 'min_samples_leaf': second}
                                 if MODEL_KIND == 'decision_tree' else
                                ({'penalty': first, 'l1_ratio': second} if MODEL_KIND in ('elastic', 'l1')
                                 else {'l2': first})))
                candidates.append({**model_params, 'threshold': threshold,
                    'smoothing_half_life': half_life, 'rebalance_every_bars': rebalance,
                    'validation': metrics})
        candidates.sort(key=lambda x: (passed(x['validation']), x['validation']['sharpe'],
                                       x['validation']['total_return']), reverse=True)
        selected = candidates[0]
        dev = prices.index[periods['training'][0]:periods['validation'][1] - HORIZON]
        if MODEL_KIND == 'rbf_svm':
            model = fit_svm_regime(features.reindex(dev), labels.reindex(dev),
                c_values=(selected['C'],), gamma_values=(selected['gamma'],), purge_gap=HORIZON)
        elif MODEL_KIND == 'decision_tree':
            model = fit_decision_tree_regime(features.reindex(dev), labels.reindex(dev),
                max_depth=selected['max_depth'], min_samples_leaf=selected['min_samples_leaf'])
        elif MODEL_KIND in ('elastic', 'l1'):
            model = fit_elastic_logistic_regime(features.reindex(dev), labels.reindex(dev),
                penalty=selected['penalty'], l1_ratio=selected['l1_ratio'])
        else:
            model = fit_logistic_regime(features.reindex(dev), labels.reindex(dev), l2=selected['l2'])
        held = prices.index[slice(*periods['held_out'])]
        scores = (predict_svm_scores(model, features.reindex(held)) if MODEL_KIND == 'rbf_svm' else
                  (predict_decision_tree_probabilities(model, features.reindex(held))
                   if MODEL_KIND == 'decision_tree' else
                  (predict_elastic_probabilities(model, features.reindex(held))
                   if MODEL_KIND in ('elastic', 'l1') else predict_regime_probabilities(model, features.reindex(held)))))
        exposure = exposure_from_scores(scores, selected['threshold'],
                                        selected['smoothing_half_life'], selected['rebalance_every_bars'])
        net = net_returns(targets(raw, assets, exposure, returns), returns).reindex(held).fillna(0)
        held_metrics = performance(net, market.reindex(held))
        if not np.isfinite(held_metrics['sharpe']) and np.allclose(net, 0.0):
            held_metrics['sharpe'] = 0.0
        runs.append({'run': number, 'periods': {k: list(v) for k, v in periods.items()},
                     'assets': assets, 'candidates_tested': len(candidates),
                     'selected': selected, 'held_out': held_metrics,
                     'held_out_passed': passed(held_metrics)})
        print(f'cycle {number}/5 complete', flush=True)
    names = ('total_return', 'sharpe', 'alpha', 'max_drawdown')
    classifier = ('gaussian_rbf_svm' if MODEL_KIND == 'rbf_svm' else
                  ('single_decision_tree' if MODEL_KIND == 'decision_tree' else
                  ('elastic_net_logistic' if MODEL_KIND == 'elastic' else
                   ('l1_logistic' if MODEL_KIND == 'l1' else 'multinomial_logistic'))))
    output = {'test': f'Non-neutral cross-sectional MV with variance-dispersion-trend {classifier}',
        'features': ['variance', 'dispersion', 'trend'], 'classifier': classifier,
        'allocation': ['cross_asset_mv', 'cash'], 'execution': {'delay_bars': 1,
        'fee': FEE, 'slippage': SLIPPAGE, 'label_horizon': HORIZON}, 'runs': runs,
        'average_held_out_metrics': {n: float(np.mean([r['held_out'][n] for r in runs])) for n in names},
        'held_out_pass_count': sum(r['held_out_passed'] for r in runs),
        'scientific_status': 'Diagnostic: these historical held-out windows were viewed earlier.'}
    path = ROOT / 'artifacts' / f'checkpoint_cmv_three_feature_{MODEL_KIND}_walkforward_summary.json'
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
