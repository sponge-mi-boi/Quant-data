"""Non-ML three-feature allocator for CMV, MT, CMT, and cash."""

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
sys.path[:0] = [str(PROJECT/'src/quant_backtester'), str(ROOT/'artifacts')]

from src.quant_backtester import get_time_period
from src.quant_backtester.hmm_regime import build_variance_dispersion_trend_features
from src.quant_backtester.market_filters_analysis import _rolling_percentile
from src.quant_backtester.strategies import (_get_signals_mv_cross_asset,
                 _get_signals_momentum_tr, _get_signals_momentum_cross_asset)
from run_cmv_full_three_stage_five_cycles import CYCLES, FEE, SLIPPAGE, performance
from run_cross_momentum_timeseries_momentum_rule_regime import normalize, net_returns, passed

FEATURE_SET=os.environ.get('RULE_FEATURE_SET','variance_dispersion_trend')


def build_correlation_liquidity_dispersion_features(prices,volume,returns,roll=20,half_life=20):
    """Causal market-level correlation, dollar-volume liquidity, and dispersion percentiles."""
    asset_mean=returns.ewm(halflife=half_life,adjust=False).mean()
    second=returns.pow(2).ewm(halflife=half_life,adjust=False).mean()
    scale=(second-asset_mean.pow(2)).clip(lower=0).pow(.5).replace(0,np.nan)
    standardized=(returns-asset_mean).div(scale)
    count=standardized.notna().sum(axis=1)
    pair_product=(standardized.sum(axis=1).pow(2)-standardized.pow(2).sum(axis=1)).div(
        (count*(count-1)).where(count>=2))
    correlation=pair_product.ewm(halflife=half_life,adjust=False).mean().clip(-1,1)
    dollar_volume=(prices*volume.reindex_like(prices)).replace(0,np.nan)
    liquidity=np.log1p(dollar_volume).median(axis=1).ewm(halflife=half_life,adjust=False).mean()
    dispersion=returns.std(axis=1,ddof=1).ewm(halflife=half_life,adjust=False).mean()
    return pd.DataFrame({'corr':_rolling_percentile(correlation,roll),
                         'liq':_rolling_percentile(liquidity,roll),
                         'dis':_rolling_percentile(dispersion,roll)}).dropna()


def allocations(features, half_life, rebalance, cmv_scale, mt_scale, cmt_scale, cash_scale):
    if FEATURE_SET=='correlation_liquidity_dispersion':
        frame=features.reindex(columns=['corr','liq','dis']).dropna()
        high_corr=((frame['corr']-.5)/.5).clip(0,1); low_corr=((.5-frame['corr'])/.5).clip(0,1)
        high_liq=((frame['liq']-.5)/.5).clip(0,1); low_liq=((.5-frame['liq'])/.5).clip(0,1)
        high_dis=((frame['dis']-.5)/.5).clip(0,1); low_dis=((.5-frame['dis'])/.5).clip(0,1)
        scores=pd.DataFrame(index=frame.index)
        scores['cross_asset_mv']=(.05+high_dis+low_corr+.3*(1-high_liq))*cmv_scale
        scores['momentum_trending']=(.05+high_corr+low_dis+.6*high_liq)*mt_scale
        scores['cross_asset_momentum_trending']=(.05+high_dis+low_corr+.8*high_liq)*cmt_scale
        scores['cash']=(.05+1.5*low_liq+.3*high_corr*high_dis)*cash_scale
        weights=scores.div(scores.sum(axis=1),axis=0).ewm(halflife=half_life,adjust=False).mean()
        weights.loc[np.arange(len(weights))%rebalance!=0]=np.nan
        return weights.ffill().fillna(0.0)
    frame=features.reindex(columns=['var','dis','trend']).dropna()
    high_var=((frame['var']-.5)/.5).clip(0,1)
    high_dis=((frame['dis']-.5)/.5).clip(0,1)
    low_dis=((.5-frame['dis'])/.5).clip(0,1)
    trend=frame['trend'].abs().clip(0,1); flat=1-trend
    scores=pd.DataFrame(index=frame.index)
    scores['cross_asset_mv']=(.05+flat+.8*high_dis)*cmv_scale
    scores['momentum_trending']=(.05+1.2*trend+.8*low_dis)*mt_scale
    scores['cross_asset_momentum_trending']=(.05+trend+.8*high_dis)*cmt_scale
    scores['cash']=(.05+1.2*high_var+.5*flat)*cash_scale
    weights=scores.div(scores.sum(axis=1),axis=0).ewm(halflife=half_life,adjust=False).mean()
    weights.loc[np.arange(len(weights))%rebalance!=0]=np.nan
    return weights.ffill().fillna(0.0)


def combine(sleeves, allocation, index):
    columns=list(dict.fromkeys(c for frame in sleeves.values() for c in frame.columns))
    target=pd.DataFrame(0.0,index=index,columns=columns)
    for name,frame in sleeves.items():
        target.loc[:,frame.columns] += frame.mul(allocation[name],axis=0)
    return target


def main():
    files={'cross_asset_mv':'checkpoint_cross_asset_mv_nonneutral_three_stage_five_cycles_summary.json',
           'momentum_trending':'checkpoint_momentum_trending_nonneutral_three_stage_five_cycles_summary.json',
           'cross_asset_momentum_trending':'checkpoint_cross_asset_momentum_trending_nonneutral_three_stage_five_cycles_summary.json'}
    sources={name:json.loads((ROOT/'artifacts'/file).read_text()) for name,file in files.items()}
    universe=pd.read_parquet(PROJECT/'data'/'close_1d_10y.parquet').columns.tolist()
    prices=get_time_period(universe,time_peri=(0,2060)); returns=prices.pct_change().fillna(0.0)
    market=get_time_period(['SPY'],time_peri=(0,2060)).reindex(prices.index)['SPY'].pct_change().fillna(0.0)
    def params(name,p): return {'stock_list':universe,'time_period':(0,2060),'freq':'d','strat_class':{name:p},'parameters_':p}
    cmv_raw=_get_signals_mv_cross_asset(params('cross_asset_mv',{'z_threshold':2.0})).reindex(prices.index).fillna(0.0)
    mt_raw=_get_signals_momentum_tr(params('momentum_trending',{'z_threshold':1.999,'roll':30}),prices).reindex(prices.index).fillna(0.0)
    cmt_raw=_get_signals_momentum_cross_asset(params('cross_asset_momentum_trending',{'z_threshold':1.9283,'roll':35})).reindex(prices.index).fillna(0.0)
    if FEATURE_SET=='correlation_liquidity_dispersion':
        volume=pd.read_parquet(PROJECT/'data'/'volume_1d_10y.parquet',columns=universe).reindex(prices.index)
        features=build_correlation_liquidity_dispersion_features(prices,volume,returns)
    else:
        features=build_variance_dispersion_trend_features(market,returns)
    runs=[]
    for number,periods in enumerate(CYCLES,1):
        sleeves={'cross_asset_mv':normalize(cmv_raw,sources['cross_asset_mv']['runs'][number-1]['validation_winner']),
                 'momentum_trending':normalize(mt_raw,sources['momentum_trending']['runs'][number-1]['validation_winner']),
                 'cross_asset_momentum_trending':normalize(cmt_raw,sources['cross_asset_momentum_trending']['runs'][number-1]['validation_winner'])}
        val=prices.index[slice(*periods['validation'])]; candidates=[]
        for half,rebalance,cmv_s,mt_s,cmt_s,cash_s in product((2,5,10),(5,10,20),(.5,1,2),(.5,1,2),(.5,1,2),(.5,1,2)):
            alloc=allocations(features,half,rebalance,cmv_s,mt_s,cmt_s,cash_s)
            metric=performance(net_returns(combine(sleeves,alloc,prices.index),returns).reindex(val).fillna(0.0),market.reindex(val))
            candidates.append({'smoothing_half_life':half,'rebalance_every_bars':rebalance,'cmv_scale':cmv_s,'mt_scale':mt_s,'cmt_scale':cmt_s,'cash_scale':cash_s,'validation':metric})
        candidates.sort(key=lambda x:(passed(x['validation']),x['validation']['sharpe'],x['validation']['total_return']),reverse=True)
        selected=candidates[0]
        alloc=allocations(features,selected['smoothing_half_life'],selected['rebalance_every_bars'],selected['cmv_scale'],selected['mt_scale'],selected['cmt_scale'],selected['cash_scale'])
        held=prices.index[slice(*periods['held_out'])]
        metric=performance(net_returns(combine(sleeves,alloc,prices.index),returns).reindex(held).fillna(0.0),market.reindex(held))
        runs.append({'run':number,'selected':selected,'average_held_out_allocations':alloc.reindex(held).mean().to_dict(),'held_out':metric,'held_out_passed':passed(metric)})
        print(f'cycle {number}/5 complete',flush=True)
    names=('total_return','sharpe','alpha','max_drawdown')
    feature_names=(['correlation','liquidity','dispersion'] if FEATURE_SET=='correlation_liquidity_dispersion' else ['variance','dispersion','trend'])
    output={'test':'Rule-regime CMV + MT + CMT + cash','features':feature_names,'liquidity_measure':'cross-sectional median log dollar volume, causal EWM percentile' if FEATURE_SET=='correlation_liquidity_dispersion' else None,'classifier':None,'allowed_sleeves':['cross_asset_mv','momentum_trending','cross_asset_momentum_trending','cash'],'validation_candidates_per_run':729,'execution':{'execution_delay_bars':1,'fee_per_order':FEE,'slippage_per_order':SLIPPAGE},'runs':runs,'average_held_out_metrics':{n:float(np.mean([r['held_out'][n] for r in runs])) for n in names},'held_out_pass_count':sum(r['held_out_passed'] for r in runs),'scientific_status':'Diagnostic: these historical windows were viewed earlier.'}
    output_name=('checkpoint_cmv_mt_cmt_corr_liq_disp_rule_regime_summary.json' if FEATURE_SET=='correlation_liquidity_dispersion' else 'checkpoint_cmv_mt_cmt_rule_regime_summary.json')
    path=ROOT/'artifacts'/output_name; path.write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8'); print(json.dumps(output,indent=2,allow_nan=False))


if __name__=='__main__': main()
