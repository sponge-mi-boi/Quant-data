"""Causal Gaussian-HMM Bayesian filter for CMV, MT, CMT, and cash."""

import json,os,sys
from itertools import combinations,product
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern,WhiteKernel

os.environ.setdefault('NUMBA_DISABLE_JIT','1')
ROOT = Path (__file__).resolve().parent.parent ; PROJECT=ROOT
sys.path[:0]=[str(PROJECT/'src/quant_backtester'),str(ROOT/'artifacts')]
from src.quant_backtester import get_time_period
from src.quant_backtester.hmm_regime import (fit_gaussian_hmm, filtered_state_probabilities,
                 probability_weighted_allocations, learned_state_allocations, describe_hmm_states,
                 build_hmm_features)
from src.quant_backtester.strategies import (_get_signals, _get_signals_mv_cross_asset,
                 _get_signals_momentum_tr, _get_signals_momentum_cross_asset)
from run_cmv_full_three_stage_five_cycles import (CYCLES,FEE,SLIPPAGE,performance,
                                                  individual_ranking)
from run_cross_momentum_timeseries_momentum_rule_regime import normalize,net_returns,passed
from run_cmv_mt_cmt_rule_regime import build_correlation_liquidity_dispersion_features

GENERAL_FILTER=os.environ.get('GENERAL_LEARNED_FILTER','false').lower()=='true'
HIGH_CONFIDENCE=os.environ.get('HIGH_CONFIDENCE_HURDLE','false').lower()=='true'
LEARNED_FEATURE_SET=os.environ.get('LEARNED_FEATURE_SET','correlation_liquidity_dispersion')
OPTIMIZE_STRATEGY_PARAMETERS=os.environ.get('OPTIMIZE_STRATEGY_PARAMETERS','false').lower()=='true'
BAYESIAN_HMM_PARAMETERS=os.environ.get('BAYESIAN_HMM_PARAMETERS','false').lower()=='true'
ROLLING_FIXED=os.environ.get('ROLLING_FIXED_WINDOWS','false').lower()=='true'
SLEEVES=(('cross_asset_mv','momentum_trending','cross_asset_momentum_trending','cointegration')
         if GENERAL_FILTER else
         ('cross_asset_mv','momentum_trending','cross_asset_momentum_trending'))
PURGE=5

def bayesian_optimize(objective,seed,initial=12,iterations=28):
    rng=np.random.default_rng(seed); x=rng.uniform(-3.,3.,size=(initial,2)); y=np.array([objective(row) for row in x])
    kernel=Matern(length_scale=np.ones(2),nu=2.5)+WhiteKernel(noise_level=1e-5)
    for _ in range(iterations):
        gp=GaussianProcessRegressor(kernel=kernel,optimizer=None,normalize_y=True,random_state=seed).fit(x,y)
        pool=rng.uniform(-4.,4.,size=(1000,2)); mu,sigma=gp.predict(pool,return_std=True); imp=mu-y.max()-.01
        z=np.divide(imp,sigma,out=np.zeros_like(imp),where=sigma>1e-12); ei=imp*norm.cdf(z)+sigma*norm.pdf(z)
        nxt=pool[int(np.argmax(ei))]; x=np.vstack((x,nxt)); y=np.append(y,objective(nxt))
    i=int(np.argmax(y)); return float(y[i]),x[i]

def decode_bayesian_hmm(theta):
    s=1/(1+np.exp(-np.clip(theta,-20,20)))
    return 1.+19.*s[0], .25+.75*s[1]

def state_mapping(probabilities,sleeve_returns,cap):
    probabilities,returns=probabilities.align(sleeve_returns,join='inner',axis=0)
    rows=[]
    for state in probabilities:
        membership=probabilities[state]; denominator=membership.sum()
        mean=returns.mul(membership,axis=0).sum()/denominator
        risk=returns.sub(mean).pow(2).mul(membership,axis=0).sum().div(denominator).pow(.5)
        score=mean.div(risk.replace(0,np.nan)).fillna(0).clip(lower=0)
        strategy=(score/score.sum() if score.sum()>0 else score).clip(upper=cap)
        row=strategy.to_dict(); row['cash']=max(0.,1-float(strategy.sum())); rows.append(row)
    return pd.DataFrame(rows,index=probabilities.columns).fillna(0.)

def combine(sleeves,allocation,index):
    columns=list(dict.fromkeys(c for frame in sleeves.values() for c in frame.columns)); target=pd.DataFrame(0.,index=index,columns=columns)
    for name,frame in sleeves.items(): target.loc[:,frame.columns]+=frame.mul(allocation[name],axis=0)
    return target

def select_strategy_on_validation(name,candidate_raw,returns,market,index,periods):
    train_index=index[slice(*periods['training'])]; val_index=index[slice(*periods['validation'])]
    validation_start,validation_stop=periods['validation']
    calculation_index=index[max(0,validation_start-2):validation_stop]
    choices=[]
    for parameters,raw in candidate_raw:
        ranked=individual_ranking(raw,returns,train_index); top=ranked.head(10).index.tolist()
        if len(top)<4: continue
        for size in range(4,len(top)+1):
            for basket in combinations(top,size):
                frame=normalize(raw.reindex(calculation_index),list(basket))
                executed=frame.shift(1).fillna(0.)
                turnover=executed.diff().abs().sum(axis=1).fillna(executed.abs().sum(axis=1))
                net=(executed*returns.reindex(calculation_index)[list(basket)]).sum(axis=1)-turnover*(FEE+SLIPPAGE)
                metric=performance(net.reindex(val_index).fillna(0.),market.reindex(val_index))
                choices.append({'parameters':parameters,'assets':list(basket),'validation':metric})
    if not choices: raise ValueError(f'no validation candidates for {name}')
    choices.sort(key=lambda x:(passed(x['validation']),x['validation']['sharpe'],
                               x['validation']['total_return']),reverse=True)
    return choices[0],len(choices)

def main():
    files={'cross_asset_mv':'checkpoint_cross_asset_mv_nonneutral_three_stage_five_cycles_summary.json','momentum_trending':'checkpoint_momentum_trending_nonneutral_three_stage_five_cycles_summary.json','cross_asset_momentum_trending':'checkpoint_cross_asset_momentum_trending_nonneutral_three_stage_five_cycles_summary.json'}
    sources={n:json.loads((ROOT/'artifacts'/f).read_text()) for n,f in files.items()}
    optimized_source=(json.loads((ROOT/'artifacts'/('checkpoint_rolling_fixed_500_260_260_three_strategy_corr_liq_disp_rbf_svm_summary.json' if ROLLING_FIXED else 'checkpoint_three_strategy_parameter_optimized_ac_vol_corr_hmm_summary.json')).read_text())
                      if BAYESIAN_HMM_PARAMETERS or ROLLING_FIXED else None)
    universe=pd.read_parquet(PROJECT/'data'/'close_1d_10y.parquet').columns.tolist(); prices=get_time_period(universe,time_peri=(0,2060)); returns=prices.pct_change().fillna(0.)
    market=get_time_period(['SPY'],time_peri=(0,2060)).reindex(prices.index)['SPY'].pct_change().fillna(0.); volume=pd.read_parquet(PROJECT/'data'/'volume_1d_10y.parquet',columns=universe).reindex(prices.index)
    if LEARNED_FEATURE_SET=='autocorrelation_volatility_correlation':
        features=build_hmm_features(market,returns).loc[:,['ac','var','corr']]
    else:
        features=build_correlation_liquidity_dispersion_features(prices,volume,returns)
    def params(n,p): return {'stock_list':universe,'time_period':(0,2060),'freq':'d','strat_class':{n:p},'parameters_':p}
    raw={'cross_asset_mv':_get_signals_mv_cross_asset(params('cross_asset_mv',{'z_threshold':2.})).reindex(prices.index).fillna(0.),'momentum_trending':_get_signals_momentum_tr(params('momentum_trending',{'z_threshold':1.999,'roll':30}),prices).reindex(prices.index).fillna(0.),'cross_asset_momentum_trending':_get_signals_momentum_cross_asset(params('cross_asset_momentum_trending',{'z_threshold':1.9283,'roll':35})).reindex(prices.index).fillna(0.)}
    parameter_candidates={}
    if OPTIMIZE_STRATEGY_PARAMETERS:
        grids={'cross_asset_mv':[{'z_threshold':z} for z in (1.5,2.,2.5)],
               'momentum_trending':[{'z_threshold':z,'roll':r} for r,z in product((20,30,60),(1.5,2.,2.5))],
               'cross_asset_momentum_trending':[{'z_threshold':z,'roll':r} for r,z in product((20,35,60),(1.5,2.,2.5))]}
        parameter_candidates={name:[] for name in grids}
        for name,grid in grids.items():
            for values in grid:
                configured=params(name,values)
                signal=(_get_signals_mv_cross_asset(configured) if name=='cross_asset_mv' else
                        (_get_signals_momentum_tr(configured,prices) if name=='momentum_trending' else
                         _get_signals_momentum_cross_asset(configured)))
                parameter_candidates[name].append((values,signal.reindex(prices.index).fillna(0.)))
    cointegration_pair=tuple(json.loads((ROOT/'artifacts'/'cointegration_one_screen_fast_summary.json').read_text())['validation_winner'])
    if GENERAL_FILTER:
        coin_params={'stock_list':list(cointegration_pair),'parameters_':{'z_threshold':1.92,'roll':32},'weights_filter':{}}
        raw['cointegration']=_get_signals(coin_params,prices[list(cointegration_pair)]).reindex(prices.index).fillna(0.)
    runs=[]
    cycles=([{'training':(260*i,500+260*i),'validation':(500+260*i,760+260*i),'held_out':(760+260*i,1020+260*i)} for i in range(5)] if ROLLING_FIXED else CYCLES)
    for number,periods in enumerate(cycles,1):
        selected_strategies={}; strategy_candidates_tested={}
        if BAYESIAN_HMM_PARAMETERS or ROLLING_FIXED:
            selected_strategies=optimized_source['runs'][number-1]['selected_strategies']
            sleeves={}
            for name in SLEEVES:
                values=selected_strategies[name]['parameters']; configured=params(name,values)
                signal=(_get_signals_mv_cross_asset(configured) if name=='cross_asset_mv' else
                        (_get_signals_momentum_tr(configured,prices) if name=='momentum_trending' else
                         _get_signals_momentum_cross_asset(configured)))
                sleeves[name]=normalize(signal.reindex(prices.index).fillna(0.),selected_strategies[name]['assets'])
        elif OPTIMIZE_STRATEGY_PARAMETERS:
            for name in SLEEVES:
                selected_strategies[name],strategy_candidates_tested[name]=select_strategy_on_validation(
                    name,parameter_candidates[name],returns,market,prices.index,periods)
            sleeves={name:normalize(
                next(frame for p,frame in parameter_candidates[name]
                     if p==selected_strategies[name]['parameters']),selected_strategies[name]['assets'])
                     for name in SLEEVES}
        else:
            sleeves={n:(normalize(raw[n],list(cointegration_pair)) if n=='cointegration' else
                        normalize(raw[n],sources[n]['runs'][number-1]['validation_winner'])) for n in SLEEVES}
        sleeve_net=pd.DataFrame({n:net_returns(f,returns) for n,f in sleeves.items()})
        train=prices.index[periods['training'][0]:periods['training'][1]-PURGE]; val=prices.index[slice(*periods['validation'])]; candidates=[]
        for states in ((2,3,4,5,6) if GENERAL_FILTER else (2,3,4)):
            model=fit_gaussian_hmm(features.reindex(train),states=states,seed=41); train_p=filtered_state_probabilities(model,features.reindex(train))
            initial=train_p.iloc[-1].to_numpy()@model['transition']; val_p=filtered_state_probabilities(model,features.reindex(val),initial)
            hurdle_grid=((2.,4.,6.) if HIGH_CONFIDENCE else
                         ((.5,1.,2.) if GENERAL_FILTER else (1.,)))
            grid=product((2,5,10),(5,10,20),(.4,.6,1.),hurdle_grid)
            if BAYESIAN_HMM_PARAMETERS:
                for rebalance in (5,10,20):
                    def objective(theta):
                        half,cap=decode_bayesian_hmm(theta); mapping=state_mapping(train_p,sleeve_net.reindex(train),cap)
                        alloc=probability_weighted_allocations(val_p,mapping,rebalance_every=rebalance,smoothing_half_life=half)
                        metric=performance(net_returns(combine(sleeves,alloc,prices.index),returns).reindex(val).fillna(0.),market.reindex(val))
                        return float(metric['sharpe']) if np.isfinite(metric['sharpe']) else -1e6
                    _,theta=bayesian_optimize(objective,number*1000+states*100+rebalance)
                    half,cap=decode_bayesian_hmm(theta); mapping=state_mapping(train_p,sleeve_net.reindex(train),cap)
                    alloc=probability_weighted_allocations(val_p,mapping,rebalance_every=rebalance,smoothing_half_life=half)
                    metric=performance(net_returns(combine(sleeves,alloc,prices.index),returns).reindex(val).fillna(0.),market.reindex(val))
                    candidates.append({'states':states,'smoothing_half_life':half,'rebalance_every_bars':rebalance,'max_sleeve_weight':cap,'confidence_hurdle':1.,'validation':metric})
                continue
            for half,rebalance,cap,hurdle in grid:
                mapping=(learned_state_allocations(train_p,sleeve_net.reindex(train),cap,hurdle)
                         if GENERAL_FILTER else state_mapping(train_p,sleeve_net.reindex(train),cap))
                alloc=probability_weighted_allocations(val_p,mapping,rebalance_every=rebalance,smoothing_half_life=half)
                metric=performance(net_returns(combine(sleeves,alloc,prices.index),returns).reindex(val).fillna(0.),market.reindex(val)); candidates.append({'states':states,'smoothing_half_life':half,'rebalance_every_bars':rebalance,'max_sleeve_weight':cap,'confidence_hurdle':hurdle,'validation':metric})
        candidates.sort(key=lambda x:(passed(x['validation']),x['validation']['sharpe'],x['validation']['total_return']),reverse=True); selected=candidates[0]
        dev=prices.index[periods['training'][0]:periods['validation'][1]-PURGE]; model=fit_gaussian_hmm(features.reindex(dev),states=selected['states'],seed=41); dev_p=filtered_state_probabilities(model,features.reindex(dev)); mapping=(learned_state_allocations(dev_p,sleeve_net.reindex(dev),selected['max_sleeve_weight'],selected['confidence_hurdle']) if GENERAL_FILTER else state_mapping(dev_p,sleeve_net.reindex(dev),selected['max_sleeve_weight']))
        held=prices.index[slice(*periods['held_out'])]; initial=dev_p.iloc[-1].to_numpy()@model['transition']; held_p=filtered_state_probabilities(model,features.reindex(held),initial); alloc=probability_weighted_allocations(held_p,mapping,rebalance_every=selected['rebalance_every_bars'],smoothing_half_life=selected['smoothing_half_life'])
        metric=performance(net_returns(combine(sleeves,alloc,prices.index),returns).reindex(held).fillna(0.),market.reindex(held)); runs.append({'run':number,'selected_strategies':selected_strategies if (OPTIMIZE_STRATEGY_PARAMETERS or BAYESIAN_HMM_PARAMETERS) else None,'strategy_validation_candidates_tested':strategy_candidates_tested if OPTIMIZE_STRATEGY_PARAMETERS else None,'selected':selected,'learned_state_feature_means':describe_hmm_states(model).to_dict(orient='index'),'state_mapping':mapping.to_dict(orient='index'),'average_held_out_allocations':alloc.reindex(held).mean().to_dict(),'held_out':metric,'held_out_passed':passed(metric)}); print(f'cycle {number}/5 complete',flush=True)
    feature_names=(['autocorrelation','volatility','correlation']
                   if LEARNED_FEATURE_SET=='autocorrelation_volatility_correlation' else
                   ['correlation','liquidity','dispersion'])
    candidate_count=(360 if BAYESIAN_HMM_PARAMETERS else (405 if GENERAL_FILTER else (243 if HIGH_CONFIDENCE else 81)))
    names=('total_return','sharpe','alpha','max_drawdown'); output={'test':'General learned-state Bayesian filter' if GENERAL_FILTER else 'Gaussian HMM Bayesian filter CMV + MT + CMT + cash','features':feature_names,'model':'causal_gaussian_hmm_filter','hyperparameter_optimizer':'Gaussian-process Bayesian optimization with expected improvement' if BAYESIAN_HMM_PARAMETERS else 'grid search','bayesian_evaluations_per_state_rebalance':40 if BAYESIAN_HMM_PARAMETERS else None,'allowed_sleeves':[*SLEEVES,'cash'],'purge_bars':PURGE,'validation_candidates_per_run':candidate_count,'execution':{'execution_delay_bars':1,'fee_per_order':FEE,'slippage_per_order':SLIPPAGE},'runs':runs,'average_held_out_metrics':{n:float(np.mean([r['held_out'][n] for r in runs])) for n in names},'held_out_pass_count':sum(r['held_out_passed'] for r in runs),'scientific_status':'Diagnostic: these historical windows were viewed earlier.'}
    output_name=('checkpoint_three_strategy_bayesian_optimized_learned_hmm_ac_vol_corr_summary.json'
                 if BAYESIAN_HMM_PARAMETERS else
                 ('checkpoint_three_strategy_parameter_optimized_ac_vol_corr_hmm_summary.json'
                 if OPTIMIZE_STRATEGY_PARAMETERS else
                 ('checkpoint_three_strategy_learned_state_filter_ac_vol_corr_high_confidence_summary.json'
                 if HIGH_CONFIDENCE and LEARNED_FEATURE_SET=='autocorrelation_volatility_correlation' and not GENERAL_FILTER else
                 ('checkpoint_general_learned_state_filter_ac_vol_corr_high_confidence_summary.json'
                 if HIGH_CONFIDENCE and LEARNED_FEATURE_SET=='autocorrelation_volatility_correlation' else
                 ('checkpoint_general_learned_state_filter_high_confidence_summary.json'
                  if HIGH_CONFIDENCE else
                 ('checkpoint_general_learned_state_filter_summary.json' if GENERAL_FILTER else
                  'checkpoint_cmv_mt_cmt_corr_liq_disp_bayesian_hmm_summary.json'))))))
    if ROLLING_FIXED:
        output_name=f'checkpoint_rolling_fixed_500_260_260_three_strategy_hmm_{LEARNED_FEATURE_SET}_{"bayesian" if BAYESIAN_HMM_PARAMETERS else "grid"}_summary.json'
    path=ROOT/'artifacts'/output_name; path.write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8'); print(json.dumps(output,indent=2,allow_nan=False))

if __name__=='__main__': main()
