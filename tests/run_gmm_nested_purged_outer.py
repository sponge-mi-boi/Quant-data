"""Nested, purged Gaussian-mixture regime allocation for one outer period."""
import json, os, sys
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault('NUMBA_DISABLE_JIT','1')
ROOT = Path (__file__).resolve().parent.parent; PROJECT=ROOT 
sys.path[:0]=[str(PROJECT/'src/quant_backtester'),str(ROOT/'artifacts')]
from src.quant_backtester import get_time_period
from src.quant_backtester.hmm_regime import probability_weighted_allocations
from src.quant_backtester.strategies import _get_signals_mv_cross_asset,_get_signals_momentum_tr,_get_signals_momentum_cross_asset
from run_cmv_full_three_stage_five_cycles import FEE,SLIPPAGE,performance
from run_cross_momentum_timeseries_momentum_rule_regime import normalize,net_returns,passed
from run_cmv_mt_cmt_rule_regime import build_correlation_liquidity_dispersion_features
from run_cmv_mt_cmt_bayesian_hmm_regime import select_strategy_on_validation,state_mapping,combine
from run_three_strategy_gmm_regime import fit_gmm,probabilities,bayesian_optimize,decode

SLEEVES=('cross_asset_mv','momentum_trending','cross_asset_momentum_trending'); PURGE=5
BASE=[{'strategy_training':(0,280),'strategy_validation':(280,400),'regime_training':(0,395),'regime_validation':(400,520)},
      {'strategy_training':(120,400),'strategy_validation':(400,520),'regime_training':(120,515),'regime_validation':(520,640)},
      {'strategy_training':(240,520),'strategy_validation':(520,640),'regime_training':(240,635),'regime_validation':(640,760)}]

def main():
    run=int(os.environ.get('NESTED_OUTER_RUN','1')); shift=260*(run-1)
    folds=[{k:(a+shift,b+shift) for k,(a,b) in f.items()} for f in BASE]; held_range=(760+shift,1020+shift)
    universe=pd.read_parquet(PROJECT/'data'/'close_1d_10y.parquet').columns.tolist(); prices=get_time_period(universe,time_peri=(0,2060)); returns=prices.pct_change().fillna(0.); market=get_time_period(['SPY'],time_peri=(0,2060)).reindex(prices.index)['SPY'].pct_change().fillna(0.)
    volume=pd.read_parquet(PROJECT/'data'/'volume_1d_10y.parquet',columns=universe).reindex(prices.index); features=build_correlation_liquidity_dispersion_features(prices,volume,returns)
    def cfg(n,p): return {'stock_list':universe,'time_period':(0,2060),'freq':'d','strat_class':{n:p},'parameters_':p}
    grids={'cross_asset_mv':[{'z_threshold':z} for z in (1.5,2.,2.5)],'momentum_trending':[{'z_threshold':z,'roll':r} for r,z in product((20,30,60),(1.5,2.,2.5))],'cross_asset_momentum_trending':[{'z_threshold':z,'roll':r} for r,z in product((20,35,60),(1.5,2.,2.5))]}
    raw={n:[] for n in SLEEVES}
    for n,g in grids.items():
        for p in g:
            c=cfg(n,p); x=(_get_signals_mv_cross_asset(c) if n=='cross_asset_mv' else (_get_signals_momentum_tr(c,prices) if n=='momentum_trending' else _get_signals_momentum_cross_asset(c)))
            raw[n].append((p,x.reindex(prices.index).fillna(0.)))
    prepared=[]
    for i,f in enumerate(folds,1):
        periods={'training':f['strategy_training'],'validation':f['strategy_validation']}; selected={}; sleeves={}
        for n in SLEEVES:
            selected[n],_=select_strategy_on_validation(n,raw[n],returns,market,prices.index,periods); frame=next(x for p,x in raw[n] if p==selected[n]['parameters']); sleeves[n]=normalize(frame,selected[n]['assets'])
        prepared.append({'spec':f,'selected':selected,'sleeves':sleeves,'net':pd.DataFrame({n:net_returns(x,returns) for n,x in sleeves.items()})}); print(f'inner strategy fold {i}/3 complete',flush=True)
    candidates=[]
    for components in (2,3,4,5,6):
      for covariance in ('full','diag'):
       fitted=[]
       for item in prepared:
        f=item['spec']; tr=prices.index[slice(*f['regime_training'])]; va=prices.index[slice(*f['regime_validation'])]; model=fit_gmm(features.reindex(tr),components,covariance); fitted.append((tr,va,probabilities(model,features.reindex(tr)),probabilities(model,features.reindex(va))))
       for rebalance in (5,10,20):
        def objective(theta):
            half,cap=decode(theta); scores=[]
            for item,(tr,va,tp,vp) in zip(prepared,fitted):
                mapping=state_mapping(tp,item['net'].reindex(tr),cap); alloc=probability_weighted_allocations(vp,mapping,rebalance_every=rebalance,smoothing_half_life=half); m=performance(net_returns(combine(item['sleeves'],alloc,prices.index),returns).reindex(va).fillna(0.),market.reindex(va)); scores.append(m['sharpe'] if np.isfinite(m['sharpe']) else -1e6)
            return float(np.median(scores))
        theta=bayesian_optimize(objective,run*10000+components*100+rebalance+(covariance=='diag')); half,cap=decode(theta); metrics=[]
        for item,(tr,va,tp,vp) in zip(prepared,fitted):
            mapping=state_mapping(tp,item['net'].reindex(tr),cap); alloc=probability_weighted_allocations(vp,mapping,rebalance_every=rebalance,smoothing_half_life=half); metrics.append(performance(net_returns(combine(item['sleeves'],alloc,prices.index),returns).reindex(va).fillna(0.),market.reindex(va)))
        candidates.append({'components':components,'covariance_type':covariance,'smoothing_half_life':half,'max_sleeve_weight':cap,'rebalance_every_bars':rebalance,'inner_fold_metrics':metrics,'median_inner_sharpe':float(np.median([m['sharpe'] for m in metrics]))})
      print(f'component {components} complete',flush=True)
    selected=max(candidates,key=lambda x:x['median_inner_sharpe']); final=prepared[-1]; dev=prices.index[shift:shift+760-PURGE]; model=fit_gmm(features.reindex(dev),selected['components'],selected['covariance_type']); dp=probabilities(model,features.reindex(dev)); mapping=state_mapping(dp,final['net'].reindex(dev),selected['max_sleeve_weight']); held=prices.index[slice(*held_range)]; hp=probabilities(model,features.reindex(held)); alloc=probability_weighted_allocations(hp,mapping,rebalance_every=selected['rebalance_every_bars'],smoothing_half_life=selected['smoothing_half_life']); metric=performance(net_returns(combine(final['sleeves'],alloc,prices.index),returns).reindex(held).fillna(0.),market.reindex(held))
    out={'test':'Gaussian mixture corr/liquidity/dispersion nested purged','outer_run':run,'features':['correlation','liquidity','dispersion'],'inner_folds':folds,'purge_bars':PURGE,'outer_held_out':list(held_range),'selected_filter':selected,'final_selected_strategies':final['selected'],'held_out':metric,'held_out_passed':passed(metric),'execution':{'execution_delay_bars':1,'fee_per_order':FEE,'slippage_per_order':SLIPPAGE},'scientific_status':'Diagnostic: this historical held-out interval was viewed earlier.'}
    path=ROOT/'artifacts'/f'checkpoint_gaussian_mixture_corr_liq_disp_nested_purged_outer_run_{run}_summary.json'; path.write_text(json.dumps(out,indent=2,allow_nan=False)); print(json.dumps(out,indent=2,allow_nan=False))
if __name__=='__main__': main()
