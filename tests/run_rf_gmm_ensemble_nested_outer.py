"""Nested ensemble of the selected RF and GMM regime allocators."""
import json, os, sys
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault('NUMBA_DISABLE_JIT','1')
ROOT = Path(r"path\to\root"); PROJECT=ROOT/'work'/'PythonProject1_basicbacktester'/'Published'
sys.path[:0]=[str(PROJECT/'src'),str(ROOT/'work')]
from src import get_time_period
from src import probability_weighted_allocations
from src import _get_signals_mv_cross_asset,_get_signals_momentum_tr,_get_signals_momentum_cross_asset
from run_cmv_full_three_stage_five_cycles import FEE,SLIPPAGE,performance
from run_cross_momentum_timeseries_momentum_rule_regime import normalize,net_returns,passed
from run_cmv_mt_cmt_rule_regime import build_correlation_liquidity_dispersion_features
from run_cmv_mt_cmt_bayesian_hmm_regime import select_strategy_on_validation,state_mapping,combine
from run_cmv_mt_cmt_logistic_regime import fit_forest_regime,predict_forest_regime,allocations,SLEEVES,HORIZON
from run_three_strategy_gmm_regime import fit_gmm,probabilities

BASE=[{'strategy_training':(0,280),'strategy_validation':(280,400),'regime_training':(0,395),'regime_validation':(400,520)}, {'strategy_training':(120,400),'strategy_validation':(400,520),'regime_training':(120,515),'regime_validation':(520,640)}, {'strategy_training':(240,520),'strategy_validation':(520,640),'regime_training':(240,635),'regime_validation':(640,760)}]
BLENDS=(0.,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.)

def labels_for(sleeves,returns,index):
    sn=pd.DataFrame({n:net_returns(x,returns) for n,x in sleeves.items()}); future=pd.DataFrame({n:(1+sn[n]).rolling(HORIZON).apply(np.prod,raw=True).sub(1).shift(-HORIZON) for n in SLEEVES}); complete=future.dropna(how='all'); winner=complete.idxmax(axis=1).reindex(future.index); best=complete.max(axis=1).reindex(future.index).fillna(0.); labels=pd.DataFrame(0.,index=index,columns=[*SLEEVES,'cash'])
    for n in SLEEVES: labels.loc[(winner==n)&(best>0),n]=1.
    labels.loc[best<=0,'cash']=1.; return labels,sn

def main():
    run=int(os.environ.get('NESTED_OUTER_RUN','1')); shift=260*(run-1); folds=[{k:(a+shift,b+shift) for k,(a,b) in f.items()} for f in BASE]; held_range=(760+shift,1020+shift)
    rf=json.loads((ROOT/'outputs'/f'checkpoint_random_forest_nested_purged_outer_run_{run}_summary.json').read_text())['selected_filter']; gm=json.loads((ROOT/'outputs'/f'checkpoint_gaussian_mixture_corr_liq_disp_nested_purged_outer_run_{run}_summary.json').read_text())['selected_filter']
    universe=pd.read_parquet(PROJECT/'data'/'processed'/'close_1d_10y.parquet').columns.tolist(); prices=get_time_period(universe,time_peri=(0,2060)); returns=prices.pct_change().fillna(0.); market=get_time_period(['SPY'],time_peri=(0,2060)).reindex(prices.index)['SPY'].pct_change().fillna(0.); volume=pd.read_parquet(PROJECT/'data'/'processed'/'volume_1d_10y.parquet',columns=universe).reindex(prices.index); features=build_correlation_liquidity_dispersion_features(prices,volume,returns)
    def cfg(n,p): return {'stock_list':universe,'time_period':(0,2060),'freq':'d','strat_class':{n:p},'parameters_':p}
    grids={'cross_asset_mv':[{'z_threshold':z} for z in (1.5,2.,2.5)],'momentum_trending':[{'z_threshold':z,'roll':r} for r,z in product((20,30,60),(1.5,2.,2.5))],'cross_asset_momentum_trending':[{'z_threshold':z,'roll':r} for r,z in product((20,35,60),(1.5,2.,2.5))]}; raw={n:[] for n in SLEEVES}
    for n,g in grids.items():
      for p in g:
        c=cfg(n,p); x=(_get_signals_mv_cross_asset(c) if n=='cross_asset_mv' else (_get_signals_momentum_tr(c,prices) if n=='momentum_trending' else _get_signals_momentum_cross_asset(c))); raw[n].append((p,x.reindex(prices.index).fillna(0.)))
    prepared=[]
    for i,f in enumerate(folds,1):
      periods={'training':f['strategy_training'],'validation':f['strategy_validation']}; selected={}; sleeves={}
      for n in SLEEVES:
        selected[n],_=select_strategy_on_validation(n,raw[n],returns,market,prices.index,periods); frame=next(x for p,x in raw[n] if p==selected[n]['parameters']); sleeves[n]=normalize(frame,selected[n]['assets'])
      labels,sn=labels_for(sleeves,returns,prices.index); prepared.append({'spec':f,'selected':selected,'sleeves':sleeves,'labels':labels,'net':sn}); print(f'inner fold {i}/3 complete',flush=True)
    fold_targets=[]
    for item in prepared:
      f=item['spec']; tr=prices.index[slice(*f['regime_training'])]; va=prices.index[slice(*f['regime_validation'])]
      rfm=fit_forest_regime(features.reindex(tr),item['labels'].reindex(tr),rf['n_estimators'],rf['max_depth'],rf['min_samples_leaf']); rfa=allocations(predict_forest_regime(rfm,features.reindex(va)),rf['smoothing_half_life'],rf['rebalance_every_bars'],rf['max_sleeve_weight']); rft=combine(item['sleeves'],rfa,prices.index)
      gmm=fit_gmm(features.reindex(tr),gm['components'],gm['covariance_type']); tp=probabilities(gmm,features.reindex(tr)); vp=probabilities(gmm,features.reindex(va)); mapping=state_mapping(tp,item['net'].reindex(tr),gm['max_sleeve_weight']); gma=probability_weighted_allocations(vp,mapping,rebalance_every=gm['rebalance_every_bars'],smoothing_half_life=gm['smoothing_half_life']); gmt=combine(item['sleeves'],gma,prices.index); fold_targets.append((va,rft,gmt))
    choices=[]
    for w in BLENDS:
      metrics=[performance(net_returns(a*w+b*(1-w),returns).reindex(va).fillna(0.),market.reindex(va)) for va,a,b in fold_targets]; choices.append({'random_forest_weight':w,'gaussian_mixture_weight':1-w,'inner_fold_metrics':metrics,'median_inner_sharpe':float(np.median([m['sharpe'] for m in metrics]))})
    selected=max(choices,key=lambda x:x['median_inner_sharpe']); final=prepared[-1]; dev=prices.index[shift:shift+760-HORIZON]; held=prices.index[slice(*held_range)]
    rfm=fit_forest_regime(features.reindex(dev),final['labels'].reindex(dev),rf['n_estimators'],rf['max_depth'],rf['min_samples_leaf']); rfa=allocations(predict_forest_regime(rfm,features.reindex(held)),rf['smoothing_half_life'],rf['rebalance_every_bars'],rf['max_sleeve_weight']); rft=combine(final['sleeves'],rfa,prices.index)
    gmm=fit_gmm(features.reindex(dev),gm['components'],gm['covariance_type']); dp=probabilities(gmm,features.reindex(dev)); hp=probabilities(gmm,features.reindex(held)); mapping=state_mapping(dp,final['net'].reindex(dev),gm['max_sleeve_weight']); gma=probability_weighted_allocations(hp,mapping,rebalance_every=gm['rebalance_every_bars'],smoothing_half_life=gm['smoothing_half_life']); gmt=combine(final['sleeves'],gma,prices.index); w=selected['random_forest_weight']; metric=performance(net_returns(rft*w+gmt*(1-w),returns).reindex(held).fillna(0.),market.reindex(held))
    out={'test':'RF + GMM corr/liquidity/dispersion nested ensemble','outer_run':run,'features':['correlation','liquidity','dispersion'],'base_filters':{'random_forest':rf,'gaussian_mixture':gm},'selected_ensemble':selected,'held_out':metric,'held_out_passed':passed(metric),'execution':{'execution_delay_bars':1,'fee_per_order':FEE,'slippage_per_order':SLIPPAGE},'scientific_status':'Diagnostic: this historical held-out interval was viewed earlier.'}; path=ROOT/'outputs'/f'checkpoint_rf_gmm_corr_liq_disp_nested_outer_run_{run}_summary.json'; path.write_text(json.dumps(out,indent=2,allow_nan=False)); print(json.dumps(out,indent=2,allow_nan=False))
if __name__=='__main__': main()
