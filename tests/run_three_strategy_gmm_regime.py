"""Walk-forward Gaussian-mixture regime allocator with Bayesian hyperparameter search."""
import json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern,WhiteKernel

os.environ.setdefault('NUMBA_DISABLE_JIT','1')
ROOT = Path(r"path\to\root"); PROJECT=ROOT/'work'/'PythonProject1_basicbacktester'/'Published'; sys.path[:0]=[str(PROJECT/'src'),str(ROOT/'work')]
from src import get_time_period
from src import probability_weighted_allocations
from src import _get_signals_mv_cross_asset,_get_signals_momentum_tr,_get_signals_momentum_cross_asset
from run_cmv_full_three_stage_five_cycles import CYCLES,FEE,SLIPPAGE,performance
from run_cross_momentum_timeseries_momentum_rule_regime import normalize,net_returns,passed
from run_cmv_mt_cmt_rule_regime import build_correlation_liquidity_dispersion_features
from run_cmv_mt_cmt_bayesian_hmm_regime import state_mapping,combine

SLEEVES=('cross_asset_mv','momentum_trending','cross_asset_momentum_trending'); PURGE=5

def fit_gmm(features,components,covariance):
    frame=features.dropna(); scaler=StandardScaler().fit(frame); model=GaussianMixture(n_components=components,covariance_type=covariance,n_init=10,reg_covar=1e-5,max_iter=500,random_state=41).fit(scaler.transform(frame))
    return {'model':model,'scaler':scaler,'columns':list(frame.columns)}

def probabilities(fitted,features):
    frame=features.reindex(columns=fitted['columns']).dropna(); values=fitted['model'].predict_proba(fitted['scaler'].transform(frame)); return pd.DataFrame(values,index=frame.index,columns=[f'state_{i}' for i in range(values.shape[1])])

def bayesian_optimize(objective,seed,initial=12,iterations=28):
    rng=np.random.default_rng(seed); x=rng.uniform(-3.,3.,size=(initial,2)); y=np.array([objective(row) for row in x]); kernel=Matern(length_scale=np.ones(2),nu=2.5)+WhiteKernel(noise_level=1e-5)
    for _ in range(iterations):
        gp=GaussianProcessRegressor(kernel=kernel,optimizer=None,normalize_y=True,random_state=seed).fit(x,y); pool=rng.uniform(-4.,4.,size=(1000,2)); mu,sigma=gp.predict(pool,return_std=True); imp=mu-y.max()-.01; z=np.divide(imp,sigma,out=np.zeros_like(imp),where=sigma>1e-12); ei=imp*norm.cdf(z)+sigma*norm.pdf(z); nxt=pool[int(np.argmax(ei))]; x=np.vstack((x,nxt)); y=np.append(y,objective(nxt))
    return x[int(np.argmax(y))]

def decode(theta):
    s=1/(1+np.exp(-np.clip(theta,-20,20))); return 1+19*s[0],.25+.75*s[1]

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
        selected_strategies=prior['selected_strategies']; sleeves={n:normalize(raw(n,selected_strategies[n]['parameters']),selected_strategies[n]['assets']) for n in SLEEVES}; sleeve_net=pd.DataFrame({n:net_returns(f,returns) for n,f in sleeves.items()}); train=prices.index[periods['training'][0]:periods['training'][1]-PURGE]; val=prices.index[slice(*periods['validation'])]; candidates=[]
        for components in (2,3,4,5,6):
            for covariance in ('full','diag'):
                fitted=fit_gmm(features.reindex(train),components,covariance); train_p=probabilities(fitted,features.reindex(train)); val_p=probabilities(fitted,features.reindex(val))
                for rebalance in (5,10,20):
                    def objective(theta):
                        half,cap=decode(theta); mapping=state_mapping(train_p,sleeve_net.reindex(train),cap); alloc=probability_weighted_allocations(val_p,mapping,rebalance_every=rebalance,smoothing_half_life=half); metric=performance(net_returns(combine(sleeves,alloc,prices.index),returns).reindex(val).fillna(0.),market.reindex(val)); return float(metric['sharpe']) if np.isfinite(metric['sharpe']) else -1e6
                    theta=bayesian_optimize(objective,number*10000+components*100+rebalance+(0 if covariance=='full' else 1)); half,cap=decode(theta); mapping=state_mapping(train_p,sleeve_net.reindex(train),cap); alloc=probability_weighted_allocations(val_p,mapping,rebalance_every=rebalance,smoothing_half_life=half); metric=performance(net_returns(combine(sleeves,alloc,prices.index),returns).reindex(val).fillna(0.),market.reindex(val)); candidates.append({'components':components,'covariance_type':covariance,'smoothing_half_life':half,'rebalance_every_bars':rebalance,'max_sleeve_weight':cap,'validation':metric})
        candidates.sort(key=lambda x:(passed(x['validation']),x['validation']['sharpe'],x['validation']['total_return']),reverse=True); selected=candidates[0]
        dev=prices.index[periods['training'][0]:periods['validation'][1]-PURGE]; fitted=fit_gmm(features.reindex(dev),selected['components'],selected['covariance_type']); dev_p=probabilities(fitted,features.reindex(dev)); mapping=state_mapping(dev_p,sleeve_net.reindex(dev),selected['max_sleeve_weight']); held=prices.index[slice(*periods['held_out'])]; held_p=probabilities(fitted,features.reindex(held)); alloc=probability_weighted_allocations(held_p,mapping,rebalance_every=selected['rebalance_every_bars'],smoothing_half_life=selected['smoothing_half_life']); metric=performance(net_returns(combine(sleeves,alloc,prices.index),returns).reindex(held).fillna(0.),market.reindex(held)); runs.append({'run':number,'selected_strategies':selected_strategies,'selected':selected,'state_feature_means':{f'state_{i}':dict(zip(features.columns,fitted['scaler'].inverse_transform(fitted['model'].means_)[i])) for i in range(selected['components'])},'state_mapping':mapping.to_dict(orient='index'),'average_held_out_allocations':alloc.reindex(held).mean().to_dict(),'held_out':metric,'held_out_passed':passed(metric)}); print(f'cycle {number}/5 complete',flush=True)
    names=('total_return','sharpe','alpha','max_drawdown'); output={'test':'Gaussian mixture learned-regime CMV + MT + CMT + cash','features':['correlation','liquidity','dispersion'],'model':'GaussianMixture','hyperparameter_optimizer':'Gaussian-process Bayesian optimization with expected improvement','component_candidates':[2,3,4,5,6],'covariance_candidates':['full','diag'],'purge_bars':PURGE,'validation_evaluations_per_run':1200,'execution':{'execution_delay_bars':1,'fee_per_order':FEE,'slippage_per_order':SLIPPAGE},'runs':runs,'average_held_out_metrics':{n:float(np.mean([r['held_out'][n] for r in runs])) for n in names},'held_out_pass_count':sum(r['held_out_passed'] for r in runs),'scientific_status':'Diagnostic: these historical windows were viewed earlier.'}; path=ROOT/'outputs'/('checkpoint_rolling_fixed_500_260_260_three_strategy_corr_liq_disp_gaussian_mixture_summary.json' if rolling else 'checkpoint_three_strategy_bayesian_optimized_corr_liq_disp_gaussian_mixture_regime_summary.json'); path.write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8'); print(json.dumps(output,indent=2,allow_nan=False))

if __name__=='__main__': main()
