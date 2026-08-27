"""Walk-forward multinomial logistic allocator for CMV, MT, CMT, and cash."""

import json
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import warnings
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern,WhiteKernel
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                             QuadraticDiscriminantAnalysis)

os.environ.setdefault('NUMBA_DISABLE_JIT', '1')
ROOT = Path(r"path\to\root")
PROJECT=ROOT/'work'/'PythonProject1_basicbacktester'/'Published'
sys.path[:0]=[str(PROJECT/'src'),str(ROOT/'work')]

from src import get_time_period
from src import (build_hmm_features,
                 build_variance_dispersion_trend_features)
from src import fit_logistic_regime,predict_regime_probabilities
from src import (fit_decision_tree_regime,
                 predict_decision_tree_probabilities)
from src import (fit_elastic_logistic_regime,
                 predict_elastic_probabilities)
from src import fit_svm_regime,predict_svm_scores
from src import (_get_signals_mv_cross_asset,
                 _get_signals_momentum_tr, _get_signals_momentum_cross_asset)
from run_cmv_full_three_stage_five_cycles import CYCLES,FEE,SLIPPAGE,performance
from run_cross_momentum_timeseries_momentum_rule_regime import normalize,net_returns,passed
from run_cmv_mt_cmt_rule_regime import build_correlation_liquidity_dispersion_features
from run_cmv_mt_cmt_bayesian_hmm_regime import select_strategy_on_validation

HORIZON=5
SLEEVES=('cross_asset_mv','momentum_trending','cross_asset_momentum_trending')
FEATURE_SET=os.environ.get('LOGISTIC_FEATURE_SET','variance_dispersion_trend')
CLASSIFIER_KIND=os.environ.get('REGIME_CLASSIFIER','logistic')
BAYESIAN_LOGISTIC=os.environ.get('BAYESIAN_LOGISTIC','false').lower()=='true'
ROLLING_FIXED=os.environ.get('ROLLING_FIXED_WINDOWS','false').lower()=='true'
ROLLING_REUSE_SELECTIONS=os.environ.get('ROLLING_REUSE_SELECTIONS','false').lower()=='true'

def bayesian_optimize(objective,seed,initial=12,iterations=28,dimensions=3):
    rng=np.random.default_rng(seed); x=rng.uniform(-3.,3.,size=(initial,dimensions)); y=np.array([objective(row) for row in x])
    kernel=Matern(length_scale=np.ones(dimensions),nu=2.5)+WhiteKernel(noise_level=1e-5)
    for _ in range(iterations):
        gp=GaussianProcessRegressor(kernel=kernel,optimizer=None,normalize_y=True,random_state=seed).fit(x,y)
        pool=rng.uniform(-4.,4.,size=(1200,dimensions)); mu,sigma=gp.predict(pool,return_std=True); imp=mu-y.max()-.01
        z=np.divide(imp,sigma,out=np.zeros_like(imp),where=sigma>1e-12); ei=imp*norm.cdf(z)+sigma*norm.pdf(z)
        nxt=pool[int(np.argmax(ei))]; x=np.vstack((x,nxt)); y=np.append(y,objective(nxt))
    i=int(np.argmax(y)); return x[i]

def decode(theta):
    s=1/(1+np.exp(-np.clip(theta,-20,20)))
    return 1+19*s[0], .25+.75*s[1], np.exp(np.log(1e-4)+(np.log(1.)-np.log(1e-4))*s[2])

def decode_elastic(theta):
    half,cap,penalty=decode(theta[:3]); ratio=1/(1+np.exp(-np.clip(theta[3],-20,20)))
    return half,cap,penalty,ratio

def decode_nn(theta):
    s=1/(1+np.exp(-np.clip(theta,-20,20))); half=1+19*s[0]; cap=.25+.75*s[1]
    units=int(np.clip(np.rint(4+60*s[2]),4,64)); alpha=np.exp(np.log(1e-6)+(np.log(.1)-np.log(1e-6))*s[3])
    return half,cap,units,alpha

def decode_tree(theta):
    s=1/(1+np.exp(-np.clip(theta,-20,20))); half=1+19*s[0]; cap=.25+.75*s[1]
    depth=int(np.clip(np.rint(2+10*s[2]),2,12)); leaf=int(np.clip(np.rint(5+75*s[3]),5,80))
    return half,cap,depth,leaf

def decode_forest(theta):
    s=1/(1+np.exp(-np.clip(theta,-20,20))); half=1+19*s[0]; cap=.25+.75*s[1]
    trees=int(np.clip(np.rint(50+150*s[2]),50,200)); depth=int(np.clip(np.rint(2+10*s[3]),2,12)); leaf=int(np.clip(np.rint(3+47*s[4]),3,50))
    return half,cap,trees,depth,leaf

def decode_svm(theta):
    s=1/(1+np.exp(-np.clip(theta,-20,20))); half=1+19*s[0]; cap=.25+.75*s[1]
    c=np.exp(np.log(.01)+(np.log(100.)-np.log(.01))*s[2]); gamma=np.exp(np.log(.001)+(np.log(10.)-np.log(.001))*s[3])
    return half,cap,c,gamma

def decode_lda(theta):
    s=1/(1+np.exp(-np.clip(theta,-20,20)))
    return 1+19*s[0], .25+.75*s[1], s[2]

def fit_lda_regime(features,labels,shrinkage):
    x,y=pd.DataFrame(features).align(pd.DataFrame(labels),join='inner',axis=0)
    valid=x.notna().all(axis=1)&y.notna().all(axis=1)
    x=x.loc[valid]; target=y.loc[valid].idxmax(axis=1)
    model=LinearDiscriminantAnalysis(solver='lsqr',shrinkage=float(shrinkage)).fit(x,target)
    return {'model':model,'columns':list(x.columns),'states':list(labels.columns)}

def predict_lda_regime(fitted,features):
    x=pd.DataFrame(features).reindex(columns=fitted['columns']).dropna()
    values=fitted['model'].predict_proba(x)
    return pd.DataFrame(values,index=x.index,columns=fitted['model'].classes_).reindex(
        columns=fitted['states'],fill_value=0.)

def fit_qda_regime(features,labels,regularization):
    x,y=pd.DataFrame(features).align(pd.DataFrame(labels),join='inner',axis=0)
    valid=x.notna().all(axis=1)&y.notna().all(axis=1)
    x=x.loc[valid]; target=y.loc[valid].idxmax(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model=QuadraticDiscriminantAnalysis(reg_param=float(regularization)).fit(x,target)
    return {'model':model,'columns':list(x.columns),'states':list(labels.columns)}

def predict_qda_regime(fitted,features):
    x=pd.DataFrame(features).reindex(columns=fitted['columns']).dropna()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        values=fitted['model'].predict_proba(x)
    return pd.DataFrame(values,index=x.index,columns=fitted['model'].classes_).reindex(
        columns=fitted['states'],fill_value=0.)

def fit_forest_regime(features,labels,trees,depth,leaf):
    x,y=pd.DataFrame(features).align(pd.DataFrame(labels),join='inner',axis=0); valid=x.notna().all(axis=1)&y.notna().all(axis=1); x=x.loc[valid]; target=y.loc[valid].idxmax(axis=1)
    model=RandomForestClassifier(n_estimators=int(trees),max_depth=int(depth),min_samples_leaf=int(leaf),max_features='sqrt',bootstrap=True,n_jobs=-1,random_state=41).fit(x,target)
    return {'model':model,'columns':list(x.columns),'states':list(labels.columns)}

def predict_forest_regime(fitted,features):
    x=pd.DataFrame(features).reindex(columns=fitted['columns']).dropna(); values=fitted['model'].predict_proba(x)
    return pd.DataFrame(values,index=x.index,columns=fitted['model'].classes_).reindex(columns=fitted['states'],fill_value=0.)

def fit_nn_regime(features,labels,units,alpha):
    x,y=pd.DataFrame(features).align(pd.DataFrame(labels),join='inner',axis=0); valid=x.notna().all(axis=1)&y.notna().all(axis=1)
    x=x.loc[valid]; target=y.loc[valid].idxmax(axis=1); mean=x.mean(); scale=x.std().replace(0,1.)
    # Numeric targets avoid sklearn's early-stopping scorer applying isnan to
    # object/string predictions. Preserve the exact regime-name mapping.
    class_names=list(pd.Index(target.unique()))
    encoded=target.map({name:i for i,name in enumerate(class_names)}).astype(int)
    model=MLPClassifier(hidden_layer_sizes=(int(units),),activation='relu',solver='adam',alpha=float(alpha),learning_rate_init=.005,max_iter=400,early_stopping=True,validation_fraction=.15,n_iter_no_change=20,tol=1e-4,random_state=41).fit((x-mean)/scale,encoded)
    return {'model':model,'mean':mean,'scale':scale,'columns':list(x.columns),'states':list(labels.columns),'class_names':class_names}

def predict_nn_regime(fitted,features):
    x=pd.DataFrame(features).reindex(columns=fitted['columns']).dropna(); values=fitted['model'].predict_proba((x-fitted['mean'])/fitted['scale'])
    names=[fitted['class_names'][int(code)] for code in fitted['model'].classes_]
    return pd.DataFrame(values,index=x.index,columns=names).reindex(columns=fitted['states'],fill_value=0.)


def allocations(probabilities,half_life,rebalance,cap):
    weights=probabilities[list(SLEEVES)].clip(upper=cap)
    weights=weights.ewm(halflife=half_life,adjust=False).mean()
    weights.loc[np.arange(len(weights))%rebalance!=0]=np.nan
    return weights.ffill().fillna(0.0)


def combine(sleeves,allocation,index):
    columns=list(dict.fromkeys(c for frame in sleeves.values() for c in frame.columns))
    target=pd.DataFrame(0.0,index=index,columns=columns)
    for name,frame in sleeves.items():
        target.loc[:,frame.columns] += frame.mul(allocation[name],axis=0)
    return target


def main():
    source_files={'cross_asset_mv':'checkpoint_cross_asset_mv_nonneutral_three_stage_five_cycles_summary.json',
      'momentum_trending':'checkpoint_momentum_trending_nonneutral_three_stage_five_cycles_summary.json',
      'cross_asset_momentum_trending':'checkpoint_cross_asset_momentum_trending_nonneutral_three_stage_five_cycles_summary.json'}
    sources={n:json.loads((ROOT/'outputs'/f).read_text()) for n,f in source_files.items()}
    optimized_source=(json.loads((ROOT/'outputs'/'checkpoint_three_strategy_parameter_optimized_ac_vol_corr_hmm_summary.json').read_text()) if BAYESIAN_LOGISTIC and not ROLLING_FIXED else None)
    rolling_source=(json.loads((ROOT/'outputs'/'checkpoint_rolling_fixed_500_260_260_three_strategy_corr_liq_disp_rbf_svm_summary.json').read_text())
                    if ROLLING_FIXED and ROLLING_REUSE_SELECTIONS else None)
    universe=pd.read_parquet(PROJECT/'data'/'processed'/'close_1d_10y.parquet').columns.tolist()
    prices=get_time_period(universe,time_peri=(0,2060)); returns=prices.pct_change().fillna(0.0)
    market=get_time_period(['SPY'],time_peri=(0,2060)).reindex(prices.index)['SPY'].pct_change().fillna(0.0)
    def params(name,p): return {'stock_list':universe,'time_period':(0,2060),'freq':'d','strat_class':{name:p},'parameters_':p}
    raw={'cross_asset_mv':_get_signals_mv_cross_asset(params('cross_asset_mv',{'z_threshold':2.0})).reindex(prices.index).fillna(0.0),
         'momentum_trending':_get_signals_momentum_tr(params('momentum_trending',{'z_threshold':1.999,'roll':30}),prices).reindex(prices.index).fillna(0.0),
         'cross_asset_momentum_trending':_get_signals_momentum_cross_asset(params('cross_asset_momentum_trending',{'z_threshold':1.9283,'roll':35})).reindex(prices.index).fillna(0.0)}
    rolling_cycles=[{'training':(260*i,500+260*i),'validation':(500+260*i,760+260*i),'held_out':(760+260*i,1020+260*i)} for i in range(5)]
    cycles=rolling_cycles if ROLLING_FIXED else CYCLES
    parameter_candidates={}
    if ROLLING_FIXED and not ROLLING_REUSE_SELECTIONS:
        grids={'cross_asset_mv':[{'z_threshold':z} for z in (1.5,2.,2.5)],
               'momentum_trending':[{'z_threshold':z,'roll':r} for r,z in product((20,30,60),(1.5,2.,2.5))],
               'cross_asset_momentum_trending':[{'z_threshold':z,'roll':r} for r,z in product((20,35,60),(1.5,2.,2.5))]}
        for name,grid in grids.items():
            parameter_candidates[name]=[]
            for values in grid:
                configured=params(name,values)
                signal=(_get_signals_mv_cross_asset(configured) if name=='cross_asset_mv' else
                        (_get_signals_momentum_tr(configured,prices) if name=='momentum_trending' else
                         _get_signals_momentum_cross_asset(configured)))
                parameter_candidates[name].append((values,signal.reindex(prices.index).fillna(0.)))
    if FEATURE_SET in ('correlation_liquidity_dispersion',
                       'correlation_liquidity_dispersion_autocorrelation',
                       'correlation_liquidity_dispersion_volatility',
                       'correlation_liquidity_volatility',
                       'correlation_dispersion'):
        volume=pd.read_parquet(PROJECT/'data'/'processed'/'volume_1d_10y.parquet',columns=universe).reindex(prices.index)
        features=build_correlation_liquidity_dispersion_features(prices,volume,returns)
        if FEATURE_SET=='correlation_liquidity_dispersion_autocorrelation':
            # Causal rolling market-return autocorrelation; build_hmm_features
            # applies the same lagged/rolling construction used by the other
            # regime experiments.  Alignment preserves the original DCL rows.
            autocorrelation=build_hmm_features(market,returns).loc[:,['ac']]
            features=features.join(autocorrelation,how='left')
        elif FEATURE_SET=='correlation_liquidity_dispersion_volatility':
            # Add the causal rolling-percentile market variance observation.
            volatility=build_hmm_features(market,returns).loc[:,['var']]
            features=features.join(volatility,how='left')
        elif FEATURE_SET=='correlation_liquidity_volatility':
            volatility=build_hmm_features(market,returns).loc[:,['var']]
            features=features.loc[:,['corr','liq']].join(volatility,how='left')
        elif FEATURE_SET=='correlation_dispersion':
            features=features.loc[:,['corr','dis']]
    else:
        features=build_variance_dispersion_trend_features(market,returns)
    runs=[]
    for number,periods in enumerate(cycles,1):
        if ROLLING_FIXED and ROLLING_REUSE_SELECTIONS:
            selected_strategies=rolling_source['runs'][number-1]['selected_strategies']; sleeves={}
            for name in SLEEVES:
                chosen=selected_strategies[name]; configured=params(name,chosen['parameters'])
                signal=(_get_signals_mv_cross_asset(configured) if name=='cross_asset_mv' else
                        (_get_signals_momentum_tr(configured,prices) if name=='momentum_trending' else
                         _get_signals_momentum_cross_asset(configured)))
                sleeves[name]=normalize(signal.reindex(prices.index).fillna(0.),chosen['assets'])
        elif ROLLING_FIXED:
            selected_strategies={}; sleeves={}
            for name in SLEEVES:
                selected_strategies[name],_=select_strategy_on_validation(
                    name,parameter_candidates[name],returns,market,prices.index,periods)
                chosen=selected_strategies[name]
                signal=next(frame for p,frame in parameter_candidates[name] if p==chosen['parameters'])
                sleeves[name]=normalize(signal,chosen['assets'])
        elif BAYESIAN_LOGISTIC:
            selected_strategies=optimized_source['runs'][number-1]['selected_strategies']; sleeves={}
            for name in SLEEVES:
                values=selected_strategies[name]['parameters']; configured=params(name,values)
                signal=(_get_signals_mv_cross_asset(configured) if name=='cross_asset_mv' else (_get_signals_momentum_tr(configured,prices) if name=='momentum_trending' else _get_signals_momentum_cross_asset(configured)))
                sleeves[name]=normalize(signal.reindex(prices.index).fillna(0.),selected_strategies[name]['assets'])
        else:
            selected_strategies=None; sleeves={name:normalize(raw[name],sources[name]['runs'][number-1]['validation_winner']) for name in SLEEVES}
        sleeve_net=pd.DataFrame({name:net_returns(frame,returns) for name,frame in sleeves.items()})
        future=pd.DataFrame({name:(1+sleeve_net[name]).rolling(HORIZON).apply(np.prod,raw=True).sub(1).shift(-HORIZON) for name in SLEEVES})
        complete=future.dropna(how='all'); winner=complete.idxmax(axis=1).reindex(future.index); best=complete.max(axis=1).reindex(future.index).fillna(0.0)
        labels=pd.DataFrame(0.0,index=prices.index,columns=[*SLEEVES,'cash'])
        for name in SLEEVES: labels.loc[(winner==name)&(best>0),name]=1.0
        labels.loc[best<=0,'cash']=1.0
        train=prices.index[periods['training'][0]:periods['training'][1]-HORIZON]
        val=prices.index[slice(*periods['validation'])]
        model_grid=(list(product((2,3,4,6),(10,30,60))) if CLASSIFIER_KIND=='decision_tree'
                    else (list(product((.001,.01,.1,1.,10.),(.25,.5,.75)))
                          if CLASSIFIER_KIND=='elastic' else
                          (list(product((.1,1.,10.),('scale',.1,1.)))
                           if CLASSIFIER_KIND=='rbf_svm' else [(None,None)])))
        candidates=[]
        if BAYESIAN_LOGISTIC:
            for rebalance in (5,10,20):
                def objective(theta):
                    if CLASSIFIER_KIND=='rbf_svm':
                        half,cap,c,gamma=decode_svm(theta); model=fit_svm_regime(features.reindex(train),labels.reindex(train),c_values=(c,),gamma_values=(gamma,),purge_gap=HORIZON); probability=predict_svm_scores(model,features.reindex(val))
                    elif CLASSIFIER_KIND=='lda':
                        half,cap,shrinkage=decode_lda(theta); model=fit_lda_regime(features.reindex(train),labels.reindex(train),shrinkage); probability=predict_lda_regime(model,features.reindex(val))
                    elif CLASSIFIER_KIND=='qda':
                        half,cap,regularization=decode_lda(theta); model=fit_qda_regime(features.reindex(train),labels.reindex(train),regularization); probability=predict_qda_regime(model,features.reindex(val))
                    elif CLASSIFIER_KIND=='random_forest':
                        half,cap,trees,depth,leaf=decode_forest(theta); model=fit_forest_regime(features.reindex(train),labels.reindex(train),trees,depth,leaf); probability=predict_forest_regime(model,features.reindex(val))
                    elif CLASSIFIER_KIND=='decision_tree':
                        half,cap,depth,leaf=decode_tree(theta); model=fit_decision_tree_regime(features.reindex(train),labels.reindex(train),max_depth=depth,min_samples_leaf=leaf); probability=predict_decision_tree_probabilities(model,features.reindex(val))
                    elif CLASSIFIER_KIND=='nn':
                        half,cap,units,alpha=decode_nn(theta); model=fit_nn_regime(features.reindex(train),labels.reindex(train),units,alpha); probability=predict_nn_regime(model,features.reindex(val))
                    elif CLASSIFIER_KIND=='elastic':
                        half,cap,penalty,ratio=decode_elastic(theta); model=fit_elastic_logistic_regime(features.reindex(train),labels.reindex(train),penalty=penalty,l1_ratio=ratio); probability=predict_elastic_probabilities(model,features.reindex(val))
                    else:
                        half,cap,l2=decode(theta); model=fit_logistic_regime(features.reindex(train),labels.reindex(train),l2=l2); probability=predict_regime_probabilities(model,features.reindex(val))
                    alloc=allocations(probability,half,rebalance,cap)
                    metric=performance(net_returns(combine(sleeves,alloc,prices.index),returns).reindex(val).fillna(0.),market.reindex(val))
                    return float(metric['sharpe']) if np.isfinite(metric['sharpe']) else -1e6
                dimensions=(5 if CLASSIFIER_KIND=='random_forest' else (4 if CLASSIFIER_KIND in ('elastic','nn','decision_tree','rbf_svm') else 3)); theta=bayesian_optimize(objective,number*1000+rebalance,dimensions=dimensions)
                if CLASSIFIER_KIND=='rbf_svm':
                    half,cap,c,gamma=decode_svm(theta); model=fit_svm_regime(features.reindex(train),labels.reindex(train),c_values=(c,),gamma_values=(gamma,),purge_gap=HORIZON); probability=predict_svm_scores(model,features.reindex(val)); regularization={'C':c,'gamma':gamma}
                elif CLASSIFIER_KIND=='lda':
                    half,cap,shrinkage=decode_lda(theta); model=fit_lda_regime(features.reindex(train),labels.reindex(train),shrinkage); probability=predict_lda_regime(model,features.reindex(val)); regularization={'shrinkage':shrinkage}
                elif CLASSIFIER_KIND=='qda':
                    half,cap,reg_param=decode_lda(theta); model=fit_qda_regime(features.reindex(train),labels.reindex(train),reg_param); probability=predict_qda_regime(model,features.reindex(val)); regularization={'reg_param':reg_param}
                elif CLASSIFIER_KIND=='random_forest':
                    half,cap,trees,depth,leaf=decode_forest(theta); model=fit_forest_regime(features.reindex(train),labels.reindex(train),trees,depth,leaf); probability=predict_forest_regime(model,features.reindex(val)); regularization={'n_estimators':trees,'max_depth':depth,'min_samples_leaf':leaf}
                elif CLASSIFIER_KIND=='decision_tree':
                    half,cap,depth,leaf=decode_tree(theta); model=fit_decision_tree_regime(features.reindex(train),labels.reindex(train),max_depth=depth,min_samples_leaf=leaf); probability=predict_decision_tree_probabilities(model,features.reindex(val)); regularization={'max_depth':depth,'min_samples_leaf':leaf}
                elif CLASSIFIER_KIND=='nn':
                    half,cap,units,alpha=decode_nn(theta); model=fit_nn_regime(features.reindex(train),labels.reindex(train),units,alpha); probability=predict_nn_regime(model,features.reindex(val)); regularization={'hidden_units':units,'alpha':alpha}
                elif CLASSIFIER_KIND=='elastic':
                    half,cap,penalty,ratio=decode_elastic(theta); model=fit_elastic_logistic_regime(features.reindex(train),labels.reindex(train),penalty=penalty,l1_ratio=ratio); probability=predict_elastic_probabilities(model,features.reindex(val)); regularization={'penalty':penalty,'l1_ratio':ratio}
                else:
                    half,cap,l2=decode(theta); model=fit_logistic_regime(features.reindex(train),labels.reindex(train),l2=l2); probability=predict_regime_probabilities(model,features.reindex(val)); regularization={'l2':l2}
                alloc=allocations(probability,half,rebalance,cap)
                metric=performance(net_returns(combine(sleeves,alloc,prices.index),returns).reindex(val).fillna(0.),market.reindex(val))
                candidates.append({**regularization,'smoothing_half_life':half,'rebalance_every_bars':rebalance,'max_sleeve_weight':cap,'validation':metric})
        for depth,leaf in model_grid:
            if BAYESIAN_LOGISTIC: break
            if CLASSIFIER_KIND=='decision_tree':
                model=fit_decision_tree_regime(features.reindex(train),labels.reindex(train),
                                               max_depth=depth,min_samples_leaf=leaf)
                probability=predict_decision_tree_probabilities(model,features.reindex(val))
            elif CLASSIFIER_KIND=='elastic':
                model=fit_elastic_logistic_regime(features.reindex(train),labels.reindex(train),
                                                  penalty=depth,l1_ratio=leaf)
                probability=predict_elastic_probabilities(model,features.reindex(val))
            elif CLASSIFIER_KIND=='rbf_svm':
                model=fit_svm_regime(features.reindex(train),labels.reindex(train),
                                     c_values=(depth,),gamma_values=(leaf,),purge_gap=HORIZON)
                probability=predict_svm_scores(model,features.reindex(val))
            else:
                model=fit_logistic_regime(features.reindex(train),labels.reindex(train),l2=0.0)
                probability=predict_regime_probabilities(model,features.reindex(val))
            for half,rebalance,cap in product((2,5,10),(5,10,20),(.4,.6,1.0)):
                alloc=allocations(probability,half,rebalance,cap)
                metric=performance(net_returns(combine(sleeves,alloc,prices.index),returns).reindex(val).fillna(0.0),market.reindex(val))
                tree_params=({'max_depth':depth,'min_samples_leaf':leaf}
                             if CLASSIFIER_KIND=='decision_tree' else
                             ({'penalty':depth,'l1_ratio':leaf}
                              if CLASSIFIER_KIND=='elastic' else
                              ({'C':depth,'gamma':leaf} if CLASSIFIER_KIND=='rbf_svm' else {})))
                candidates.append({**tree_params,'smoothing_half_life':half,'rebalance_every_bars':rebalance,'max_sleeve_weight':cap,'validation':metric})
        candidates.sort(key=lambda x:(passed(x['validation']),x['validation']['sharpe'],x['validation']['total_return']),reverse=True)
        selected=candidates[0]
        dev=prices.index[periods['training'][0]:periods['validation'][1]-HORIZON]
        if CLASSIFIER_KIND=='random_forest':
            model=fit_forest_regime(features.reindex(dev),labels.reindex(dev),selected['n_estimators'],selected['max_depth'],selected['min_samples_leaf'])
        elif CLASSIFIER_KIND=='nn':
            model=fit_nn_regime(features.reindex(dev),labels.reindex(dev),selected['hidden_units'],selected['alpha'])
        elif CLASSIFIER_KIND=='decision_tree':
            model=fit_decision_tree_regime(features.reindex(dev),labels.reindex(dev),
                                           max_depth=selected['max_depth'],
                                           min_samples_leaf=selected['min_samples_leaf'])
        elif CLASSIFIER_KIND=='elastic':
            model=fit_elastic_logistic_regime(features.reindex(dev),labels.reindex(dev),
                                              penalty=selected['penalty'],
                                              l1_ratio=selected['l1_ratio'])
        elif CLASSIFIER_KIND=='rbf_svm':
            model=fit_svm_regime(features.reindex(dev),labels.reindex(dev),
                                 c_values=(selected['C'],),gamma_values=(selected['gamma'],),
                                 purge_gap=HORIZON)
        elif CLASSIFIER_KIND=='lda':
            model=fit_lda_regime(features.reindex(dev),labels.reindex(dev),selected['shrinkage'])
        elif CLASSIFIER_KIND=='qda':
            model=fit_qda_regime(features.reindex(dev),labels.reindex(dev),selected['reg_param'])
        else:
            model=fit_logistic_regime(features.reindex(dev),labels.reindex(dev),l2=selected.get('l2',0.0))
        held=prices.index[slice(*periods['held_out'])]
        probability=(predict_forest_regime(model,features.reindex(held)) if CLASSIFIER_KIND=='random_forest' else
                     (predict_qda_regime(model,features.reindex(held)) if CLASSIFIER_KIND=='qda' else
                     (predict_lda_regime(model,features.reindex(held)) if CLASSIFIER_KIND=='lda' else
                     (predict_nn_regime(model,features.reindex(held)) if CLASSIFIER_KIND=='nn' else
                     (predict_decision_tree_probabilities(model,features.reindex(held))
                     if CLASSIFIER_KIND=='decision_tree' else
                     (predict_elastic_probabilities(model,features.reindex(held))
                      if CLASSIFIER_KIND=='elastic' else
                      (predict_svm_scores(model,features.reindex(held))
                       if CLASSIFIER_KIND=='rbf_svm' else
                       predict_regime_probabilities(model,features.reindex(held)))))))))
        alloc=allocations(probability,selected['smoothing_half_life'],selected['rebalance_every_bars'],selected['max_sleeve_weight'])
        metric=performance(net_returns(combine(sleeves,alloc,prices.index),returns).reindex(held).fillna(0.0),market.reindex(held))
        average=alloc.reindex(held).mean().to_dict(); average['cash']=float(1-alloc.reindex(held).sum(axis=1).mean())
        runs.append({'run':number,'selected_strategies':selected_strategies,'selected':selected,'average_held_out_allocations':average,'held_out':metric,'held_out_passed':passed(metric)})
        print(f'cycle {number}/5 complete',flush=True)
    names=('total_return','sharpe','alpha','max_drawdown')
    feature_names=(['correlation','liquidity','dispersion','autocorrelation']
                   if FEATURE_SET=='correlation_liquidity_dispersion_autocorrelation' else
                   (['correlation','liquidity','dispersion','volatility']
                    if FEATURE_SET=='correlation_liquidity_dispersion_volatility' else
                   (['correlation','liquidity','volatility']
                    if FEATURE_SET=='correlation_liquidity_volatility' else
                   (['correlation','dispersion']
                    if FEATURE_SET=='correlation_dispersion' else
                   (['correlation','liquidity','dispersion']
                    if FEATURE_SET=='correlation_liquidity_dispersion' else
                    ['variance','dispersion','trend'])))))
    classifier_name=('gaussian_rbf_svm_bayesian_optimized' if BAYESIAN_LOGISTIC and CLASSIFIER_KIND=='rbf_svm' else
                     ('regularized_qda_bayesian_optimized' if BAYESIAN_LOGISTIC and CLASSIFIER_KIND=='qda' else
                     ('shrinkage_lda_bayesian_optimized' if BAYESIAN_LOGISTIC and CLASSIFIER_KIND=='lda' else
                     ('random_forest_bayesian_optimized' if BAYESIAN_LOGISTIC and CLASSIFIER_KIND=='random_forest' else
                     ('decision_tree_bayesian_optimized' if BAYESIAN_LOGISTIC and CLASSIFIER_KIND=='decision_tree' else
                     ('single_hidden_layer_neural_network_bayesian_optimized' if BAYESIAN_LOGISTIC and CLASSIFIER_KIND=='nn' else
                     ('elastic_net_logistic_bayesian_optimized' if BAYESIAN_LOGISTIC and CLASSIFIER_KIND=='elastic' else
                     ('multinomial_logistic_l2_optimized' if BAYESIAN_LOGISTIC else
                     ('decision_tree' if CLASSIFIER_KIND=='decision_tree' else
                     ('elastic_net_logistic' if CLASSIFIER_KIND=='elastic' else
                      ('gaussian_rbf_svm' if CLASSIFIER_KIND=='rbf_svm' else
                       'multinomial_logistic_unregularized')))))))))))
    candidate_count=(324 if CLASSIFIER_KIND=='decision_tree' else
                     (405 if CLASSIFIER_KIND=='elastic' else 27))
    if CLASSIFIER_KIND=='rbf_svm': candidate_count=243
    if BAYESIAN_LOGISTIC: candidate_count=120
    output={'test':f'{classifier_name} CMV + MT + CMT + cash','walk_forward':({'type':'fixed_rolling','training_bars':500,'validation_bars':260,'held_out_bars':260,'step_bars':260,'full_reset_each_run':True} if ROLLING_FIXED else {'type':'expanding'}),'features':feature_names,'liquidity_measure':'cross-sectional median log dollar volume, causal EWM percentile' if FEATURE_SET=='correlation_liquidity_dispersion' else None,'classifier':classifier_name,'hyperparameter_optimizer':'Gaussian-process Bayesian optimization with expected improvement' if BAYESIAN_LOGISTIC else 'grid search','target_horizon_bars':HORIZON,'purge_bars':HORIZON,'allowed_sleeves':[*SLEEVES,'cash'],'validation_candidates_per_run':candidate_count,'execution':{'execution_delay_bars':1,'fee_per_order':FEE,'slippage_per_order':SLIPPAGE},'runs':runs,'average_held_out_metrics':{n:float(np.mean([r['held_out'][n] for r in runs])) for n in names},'held_out_pass_count':sum(r['held_out_passed'] for r in runs),'scientific_status':'Diagnostic: these historical windows were viewed earlier.'}
    if ROLLING_FIXED:
        feature_slug={'correlation_liquidity_dispersion':'corr_liq_disp','correlation_dispersion':'corr_disp','correlation_liquidity_dispersion_autocorrelation':'corr_liq_disp_ac','correlation_liquidity_dispersion_volatility':'corr_liq_disp_vol','correlation_liquidity_volatility':'corr_liq_vol'}.get(FEATURE_SET,'var_disp_trend')
        output_name=f'checkpoint_rolling_fixed_500_260_260_three_strategy_{feature_slug}_{classifier_name}_summary.json'
    elif CLASSIFIER_KIND=='rbf_svm' and BAYESIAN_LOGISTIC:
        output_name=('checkpoint_three_strategy_bayesian_optimized_corr_liq_disp_ac_rbf_svm_regime_summary.json'
                     if FEATURE_SET=='correlation_liquidity_dispersion_autocorrelation' else
                     ('checkpoint_three_strategy_bayesian_optimized_corr_liq_disp_vol_rbf_svm_regime_summary.json'
                      if FEATURE_SET=='correlation_liquidity_dispersion_volatility' else
                      ('checkpoint_three_strategy_bayesian_optimized_corr_liq_vol_rbf_svm_regime_summary.json'
                       if FEATURE_SET=='correlation_liquidity_volatility' else
                       ('checkpoint_three_strategy_bayesian_optimized_corr_disp_rbf_svm_regime_summary.json'
                        if FEATURE_SET=='correlation_dispersion' else
                        'checkpoint_three_strategy_bayesian_optimized_corr_liq_disp_rbf_svm_regime_summary.json'))))
    elif CLASSIFIER_KIND=='lda' and BAYESIAN_LOGISTIC:
        output_name='checkpoint_three_strategy_bayesian_optimized_corr_liq_disp_lda_regime_summary.json'
    elif CLASSIFIER_KIND=='qda' and BAYESIAN_LOGISTIC:
        output_name='checkpoint_three_strategy_bayesian_optimized_corr_liq_disp_qda_regime_summary.json'
    elif CLASSIFIER_KIND=='random_forest':
        output_name='checkpoint_three_strategy_bayesian_optimized_corr_liq_disp_random_forest_regime_summary.json'
    elif CLASSIFIER_KIND=='decision_tree' and BAYESIAN_LOGISTIC:
        output_name='checkpoint_three_strategy_bayesian_optimized_corr_liq_disp_decision_tree_regime_summary.json'
    elif CLASSIFIER_KIND=='nn':
        output_name='checkpoint_three_strategy_bayesian_optimized_corr_liq_disp_neural_network_regime_summary.json'
    elif CLASSIFIER_KIND=='decision_tree':
        output_name='checkpoint_cmv_mt_cmt_corr_liq_disp_decision_tree_regime_summary.json'
    elif CLASSIFIER_KIND=='elastic' and BAYESIAN_LOGISTIC:
        output_name='checkpoint_three_strategy_bayesian_optimized_corr_liq_disp_elastic_logistic_regime_summary.json'
    elif CLASSIFIER_KIND=='elastic':
        output_name='checkpoint_cmv_mt_cmt_corr_liq_disp_elastic_regime_summary.json'
    elif CLASSIFIER_KIND=='rbf_svm':
        output_name='checkpoint_cmv_mt_cmt_corr_liq_disp_rbf_svm_regime_summary.json'
    elif BAYESIAN_LOGISTIC:
        output_name='checkpoint_three_strategy_bayesian_optimized_corr_liq_disp_logistic_regime_summary.json'
    else:
        output_name=('checkpoint_cmv_mt_cmt_corr_liq_disp_logistic_regime_summary.json' if FEATURE_SET=='correlation_liquidity_dispersion' else 'checkpoint_cmv_mt_cmt_logistic_regime_summary.json')
    path=ROOT/'outputs'/output_name; path.write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8'); print(json.dumps(output,indent=2,allow_nan=False))


if __name__=='__main__': main()
