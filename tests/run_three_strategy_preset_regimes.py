"""Three preset correlation/liquidity/dispersion regimes for optimized CMV, MT and CMT."""
import json,os,sys
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
os.environ.setdefault('NUMBA_DISABLE_JIT','1')
ROOT = Path(r"path\to\root"); PROJECT=ROOT/'work'/'PythonProject1_basicbacktester'/'Published'; sys.path[:0]=[str(PROJECT/'src'),str(ROOT/'work')]
from src import get_time_period
from run_cmv_mt_cmt_rule_regime import build_correlation_liquidity_dispersion_features
from src import _get_signals_mv_cross_asset,_get_signals_momentum_tr,_get_signals_momentum_cross_asset
from run_cmv_full_three_stage_five_cycles import CYCLES,FEE,SLIPPAGE,performance
from run_cross_momentum_timeseries_momentum_rule_regime import normalize,net_returns,passed
SLEEVES=('cross_asset_mv','momentum_trending','cross_asset_momentum_trending')

def allocations(features,half,rebalance,cmv_scale,mt_scale,cmt_scale,cash_scale):
    f=features.reindex(columns=['corr','liq','dis']).dropna()
    high_corr=((f['corr']-.5)/.5).clip(0,1); low_corr=((.5-f['corr'])/.5).clip(0,1)
    high_liq=((f['liq']-.5)/.5).clip(0,1); low_liq=((.5-f['liq'])/.5).clip(0,1)
    high_dis=((f['dis']-.5)/.5).clip(0,1); low_dis=((.5-f['dis'])/.5).clip(0,1)
    score=pd.DataFrame(index=f.index)
    score['cross_asset_mv']=(.05+high_dis+low_corr+.3*(1-high_liq))*cmv_scale
    score['momentum_trending']=(.05+high_corr+low_dis+.6*high_liq)*mt_scale
    score['cross_asset_momentum_trending']=(.05+high_dis+low_corr+.8*high_liq)*cmt_scale
    score['cash']=(.05+1.5*low_liq+.3*high_corr*high_dis)*cash_scale
    weight=score.div(score.sum(axis=1),axis=0).ewm(halflife=half,adjust=False).mean(); weight.loc[np.arange(len(weight))%rebalance!=0]=np.nan
    return weight.ffill().fillna(0.)

def combine(sleeves,allocation,index):
    cols=list(dict.fromkeys(c for f in sleeves.values() for c in f.columns)); target=pd.DataFrame(0.,index=index,columns=cols)
    for name,frame in sleeves.items(): target.loc[:,frame.columns]+=frame.mul(allocation[name],axis=0)
    return target

def main():
    rolling=os.environ.get('ROLLING_FIXED_WINDOWS','false').lower()=='true'; source=json.loads((ROOT/'outputs'/('checkpoint_rolling_fixed_500_260_260_three_strategy_corr_liq_disp_rbf_svm_summary.json' if rolling else 'checkpoint_three_strategy_parameter_optimized_ac_vol_corr_hmm_summary.json')).read_text()); universe=pd.read_parquet(PROJECT/'data'/'processed'/'close_1d_10y.parquet').columns.tolist(); prices=get_time_period(universe,time_peri=(0,2060)); returns=prices.pct_change().fillna(0.); market=get_time_period(['SPY'],time_peri=(0,2060)).reindex(prices.index)['SPY'].pct_change().fillna(0.); volume=pd.read_parquet(PROJECT/'data'/'processed'/'volume_1d_10y.parquet',columns=universe).reindex(prices.index); features=build_correlation_liquidity_dispersion_features(prices,volume,returns)
    cache={}; runs=[]
    def raw(name,p):
        key=(name,tuple(sorted(p.items())))
        if key not in cache:
            cfg={'stock_list':universe,'time_period':(0,2060),'freq':'d','strat_class':{name:p},'parameters_':p}; cache[key]=(_get_signals_mv_cross_asset(cfg) if name=='cross_asset_mv' else (_get_signals_momentum_tr(cfg,prices) if name=='momentum_trending' else _get_signals_momentum_cross_asset(cfg))).reindex(prices.index).fillna(0.)
        return cache[key]
    cycles=([{'training':(260*i,500+260*i),'validation':(500+260*i,760+260*i),'held_out':(760+260*i,1020+260*i)} for i in range(5)] if rolling else CYCLES)
    for number,(periods,prior) in enumerate(zip(cycles,source['runs']),1):
        selected_strategies=prior['selected_strategies']; sleeves={n:normalize(raw(n,selected_strategies[n]['parameters']),selected_strategies[n]['assets']) for n in SLEEVES}; val=prices.index[slice(*periods['validation'])]; candidates=[]
        for half,rebalance,cmv_s,mt_s,cmt_s,cash_s in product((2,5,10),(5,10,20),(.5,1,2),(.5,1,2),(.5,1,2),(.5,1,2)):
            alloc=allocations(features,half,rebalance,cmv_s,mt_s,cmt_s,cash_s); metric=performance(net_returns(combine(sleeves,alloc,prices.index),returns).reindex(val).fillna(0.),market.reindex(val)); candidates.append({'smoothing_half_life':half,'rebalance_every_bars':rebalance,'cmv_scale':cmv_s,'mt_scale':mt_s,'cmt_scale':cmt_s,'cash_scale':cash_s,'validation':metric})
        candidates.sort(key=lambda x:(passed(x['validation']),x['validation']['sharpe'],x['validation']['total_return']),reverse=True); selected=candidates[0]; alloc=allocations(features,selected['smoothing_half_life'],selected['rebalance_every_bars'],selected['cmv_scale'],selected['mt_scale'],selected['cmt_scale'],selected['cash_scale']); held=prices.index[slice(*periods['held_out'])]; metric=performance(net_returns(combine(sleeves,alloc,prices.index),returns).reindex(held).fillna(0.),market.reindex(held)); runs.append({'run':number,'selected':selected,'average_held_out_allocations':alloc.reindex(held).mean().to_dict(),'held_out':metric,'held_out_passed':passed(metric)}); print(f'cycle {number}/5 complete',flush=True)
    names=('total_return','sharpe','alpha','max_drawdown'); output={'test':'Parameter-optimized CMV + MT + CMT with three preset regimes','features':['dispersion','correlation','liquidity'],'liquidity_measure':'cross-sectional median log dollar volume, causal EWM percentile','regimes':['cross_asset_mean_reversion','broad_momentum','cross_sectional_momentum'],'classifier':None,'validation_candidates_per_run':729,'execution':{'execution_delay_bars':1,'fee_per_order':FEE,'slippage_per_order':SLIPPAGE},'runs':runs,'average_held_out_metrics':{n:float(np.mean([r['held_out'][n] for r in runs])) for n in names},'held_out_pass_count':sum(r['held_out_passed'] for r in runs),'scientific_status':'Diagnostic: these historical windows were viewed earlier.'}; path=ROOT/'outputs'/('checkpoint_rolling_fixed_500_260_260_three_strategy_disp_corr_liq_preset_grid_summary.json' if rolling else 'checkpoint_three_strategy_parameter_optimized_disp_corr_liq_preset_regimes_summary.json'); path.write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8'); print(json.dumps(output,indent=2,allow_nan=False))
if __name__=='__main__': main()
