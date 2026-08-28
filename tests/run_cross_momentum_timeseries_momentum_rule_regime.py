"""Non-ML variance/dispersion/trend allocator for two momentum sleeves and cash."""

import json
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault('NUMBA_DISABLE_JIT', '1')
ROOT = Path (__file__).resolve().parent.parent 

PROJECT = ROOT 
sys.path[:0] = [str(PROJECT / 'src'/'quant_backtester'), str(ROOT / 'tests')]

from src.quant_backtester import get_time_period
from src.quant_backtester.hmm_regime import build_variance_dispersion_trend_features
from src.quant_backtester.strategies import (_get_signals_momentum_cross_asset,
                 _get_signals_momentum_tr,
                 _get_signals_mv_cross_asset)
from run_cmv_full_three_stage_five_cycles import CYCLES, FEE, SLIPPAGE, performance

PAIR_KIND = os.environ.get('RULE_REGIME_PAIR', 'cross_momentum')
PRIMARY = ('cross_asset_mv' if PAIR_KIND == 'cmv' else 'cross_asset_momentum_trending')


def normalize(raw, assets):
    frame = raw[assets].copy()
    return frame.div(frame.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def rule_allocations(features, half_life, rebalance, cross_scale, time_scale, cash_scale):
    frame = features.reindex(columns=['var', 'dis', 'trend']).dropna()
    high_var = ((frame['var'] - .5) / .5).clip(0, 1)
    high_dis = ((frame['dis'] - .5) / .5).clip(0, 1)
    low_dis = ((.5 - frame['dis']) / .5).clip(0, 1)
    trend = frame['trend'].abs().clip(0, 1)
    scores = pd.DataFrame(index=frame.index)
    if PAIR_KIND == 'cmv':
        scores[PRIMARY] = (.05 + (1 - trend) + .8 * high_dis) * cross_scale
    else:
        scores[PRIMARY] = (.05 + trend + .8 * high_dis) * cross_scale
    scores['momentum_trending'] = (.05 + 1.2 * trend + .8 * low_dis) * time_scale
    scores['cash'] = (.05 + 1.2 * high_var + .5 * (1 - trend)) * cash_scale
    weights = scores.div(scores.sum(axis=1), axis=0)
    weights = weights.ewm(halflife=half_life, adjust=False).mean()
    weights.loc[np.arange(len(weights)) % rebalance != 0] = np.nan
    return weights.ffill().fillna(0.0)


def combine(cross, time, allocation):
    columns = list(dict.fromkeys([*cross.columns, *time.columns]))
    target = pd.DataFrame(0.0, index=cross.index, columns=columns)
    target.loc[:, cross.columns] += cross.mul(allocation[PRIMARY], axis=0)
    target.loc[:, time.columns] += time.mul(allocation['momentum_trending'], axis=0)
    return target


def net_returns(target, returns):
    executed = target.shift(1).fillna(0.0)
    turnover = executed.diff().abs().sum(axis=1).fillna(executed.abs().sum(axis=1))
    return (executed * returns[target.columns]).sum(axis=1) - turnover * (FEE + SLIPPAGE)


def passed(metrics):
    return bool(np.isfinite(metrics['sharpe']) and metrics['total_return'] > 0 and metrics['sharpe'] > 0)


def main():
    primary_file = ('checkpoint_cross_asset_mv_nonneutral_three_stage_five_cycles_summary.json'
                    if PAIR_KIND == 'cmv' else
                    'checkpoint_cross_asset_momentum_trending_nonneutral_three_stage_five_cycles_summary.json')
    cross_source = json.loads((ROOT/'artifacts'/primary_file).read_text())
    time_source = json.loads((ROOT/'artifacts'/'checkpoint_momentum_trending_nonneutral_three_stage_five_cycles_summary.json').read_text())
    universe = pd.read_parquet(PROJECT/'data'/'close_1d_10y.parquet').columns.tolist()
    prices = get_time_period(universe, time_peri=(0, 2060)); returns = prices.pct_change().fillna(0.0)
    market = get_time_period(['SPY'], time_peri=(0, 2060)).reindex(prices.index)['SPY'].pct_change().fillna(0.0)
    primary_parameters=({'z_threshold':2.0} if PAIR_KIND == 'cmv'
                        else {'z_threshold':1.9283,'roll':35})
    cross_params={'stock_list':universe,'time_period':(0,2060),'freq':'d',
                  'strat_class':{PRIMARY:primary_parameters},'parameters_':primary_parameters}
    time_params={'stock_list':universe,'time_period':(0,2060),'freq':'d','strat_class':{'momentum_trending':{'z_threshold':1.999,'roll':30}},'parameters_':{'z_threshold':1.999,'roll':30}}
    cross_raw=(_get_signals_mv_cross_asset(cross_params) if PAIR_KIND == 'cmv'
               else _get_signals_momentum_cross_asset(cross_params)).reindex(prices.index).fillna(0.0)
    time_raw=_get_signals_momentum_tr(time_params,prices).reindex(prices.index).fillna(0.0)
    features=build_variance_dispersion_trend_features(market,returns)
    runs=[]
    for number,(periods,cr,tr) in enumerate(zip(CYCLES,cross_source['runs'],time_source['runs']),1):
        cross=normalize(cross_raw,cr['validation_winner']); time=normalize(time_raw,tr['validation_winner'])
        val=prices.index[slice(*periods['validation'])]; candidates=[]
        for half,rebalance,cross_scale,time_scale,cash_scale in product((2,5,10),(5,10,20),(.5,1,2),(.5,1,2),(.5,1,2)):
            allocation=rule_allocations(features,half,rebalance,cross_scale,time_scale,cash_scale)
            metrics=performance(net_returns(combine(cross,time,allocation),returns).reindex(val).fillna(0.0),market.reindex(val))
            candidates.append({'smoothing_half_life':half,'rebalance_every_bars':rebalance,'cross_scale':cross_scale,'time_scale':time_scale,'cash_scale':cash_scale,'validation':metrics})
        candidates.sort(key=lambda x:(passed(x['validation']),x['validation']['sharpe'],x['validation']['total_return']),reverse=True)
        selected=candidates[0]
        allocation=rule_allocations(features,selected['smoothing_half_life'],selected['rebalance_every_bars'],selected['cross_scale'],selected['time_scale'],selected['cash_scale'])
        held=prices.index[slice(*periods['held_out'])]
        metrics=performance(net_returns(combine(cross,time,allocation),returns).reindex(held).fillna(0.0),market.reindex(held))
        runs.append({'run':number,'selected':selected,'average_held_out_allocations':allocation.reindex(held).mean().to_dict(),'held_out':metrics,'held_out_passed':passed(metrics)})
        print(f'cycle {number}/5 complete',flush=True)
    names=('total_return','sharpe','alpha','max_drawdown')
    pair_label=('CMV' if PAIR_KIND == 'cmv' else 'cross-sectional momentum')
    output={'test':f'Rule-regime {pair_label} + time-series momentum + cash','features':['variance','dispersion','trend'],'classifier':None,'allowed_sleeves':[PRIMARY,'momentum_trending','cash'],'validation_candidates_per_run':243,'execution':{'execution_delay_bars':1,'fee_per_order':FEE,'slippage_per_order':SLIPPAGE},'runs':runs,'average_held_out_metrics':{n:float(np.mean([r['held_out'][n] for r in runs])) for n in names},'held_out_pass_count':sum(r['held_out_passed'] for r in runs),'scientific_status':'Diagnostic: these historical windows were viewed earlier.'}
    output_name=('checkpoint_cmv_timeseries_momentum_rule_regime_summary.json' if PAIR_KIND == 'cmv'
                 else 'checkpoint_cross_momentum_timeseries_momentum_rule_regime_summary.json')
    path=ROOT/'artifacts'/output_name
    path.write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8'); print(json.dumps(output,indent=2,allow_nan=False))


if __name__=='__main__': main()
