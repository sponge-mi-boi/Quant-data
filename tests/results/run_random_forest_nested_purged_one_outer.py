"""One outer random-forest test with nested chronological purged validation."""
import json, os, sys
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd

os.environ.setdefault('NUMBA_DISABLE_JIT','1')
ROOT  = Path(r"path\to\root")
PROJECT=ROOT/'work'/'PythonProject1_basicbacktester'/'Published'
sys.path[:0]=[str(PROJECT/'src'),str(ROOT/'work')]
MODEL_KIND=os.environ.get('NESTED_MODEL_KIND','random_forest')
OUTER_RUN=int(os.environ.get('NESTED_OUTER_RUN','1'))
FEATURE_SET=os.environ.get('NESTED_FEATURE_SET','correlation_liquidity_dispersion')
NEUTRALITY_MODE=os.environ.get('NESTED_NEUTRALITY','none').lower()
UNIVERSE_FILTER=os.environ.get('NESTED_UNIVERSE_FILTER','none').lower()
STRATEGY_SET=os.environ.get('NESTED_STRATEGY_SET','three').lower()
VALID_MODELS={'random_forest','decision_tree','elastic_net_logistic','rbf_svm','neural_network','transformer'}
VALID_FEATURE_SETS={'correlation_liquidity_dispersion','autocorrelation_correlation_volatility'}
if MODEL_KIND not in VALID_MODELS:
    raise ValueError(f'NESTED_MODEL_KIND must be one of {sorted(VALID_MODELS)}; got {MODEL_KIND!r}')
if FEATURE_SET not in VALID_FEATURE_SETS:
    raise ValueError(f'NESTED_FEATURE_SET must be one of {sorted(VALID_FEATURE_SETS)}; got {FEATURE_SET!r}')
if OUTER_RUN not in range(1,6):
    raise ValueError(f'NESTED_OUTER_RUN must be between 1 and 5; got {OUTER_RUN}')
if NEUTRALITY_MODE not in {'none','beta','pc'}:
    raise ValueError(f'NESTED_NEUTRALITY must be none, beta, or pc; got {NEUTRALITY_MODE!r}')
if UNIVERSE_FILTER not in {'none','historical_large','historical_mega'}:
    raise ValueError('NESTED_UNIVERSE_FILTER must be none, historical_large, or historical_mega')
if STRATEGY_SET not in {'three','time_series_momentum'}:
    raise ValueError('NESTED_STRATEGY_SET must be three or time_series_momentum')
if NEUTRALITY_MODE!='none':
    os.environ['NEUTRALITY_MODE']=NEUTRALITY_MODE

from src import get_time_period
from src import (_get_signals_mv_cross_asset,
                 _get_signals_momentum_tr, _get_signals_momentum_cross_asset)
from run_cmv_full_three_stage_five_cycles import FEE,SLIPPAGE,performance
from run_cross_momentum_timeseries_momentum_rule_regime import normalize,net_returns,passed
from run_cmv_mt_cmt_rule_regime import build_correlation_liquidity_dispersion_features
from run_cmv_mt_cmt_bayesian_hmm_regime import select_strategy_on_validation
from run_cmv_mt_cmt_logistic_regime import (SLEEVES,HORIZON,fit_forest_regime,
    predict_forest_regime,allocations,combine,bayesian_optimize,decode_forest,decode_tree,
    decode_elastic,decode_svm,decode_nn,fit_nn_regime,predict_nn_regime)
from src import (fit_decision_tree_regime,
                 predict_decision_tree_probabilities)
from src import (fit_elastic_logistic_regime,
                 predict_elastic_probabilities)
from src import fit_svm_regime,predict_svm_scores
if MODEL_KIND=='transformer':
    from transformer_regime import fit_transformer,predict_transformer
from src import build_hmm_features
from run_three_strategy_adam_disp_corr_liq import neutrality_cache,neutralize,neutrality_residuals

if STRATEGY_SET=='time_series_momentum':
    SLEEVES=('momentum_trending',)

# Each inner fold separates strategy selection from regime validation.
FOLDS=[
 {'strategy_training':(0,280),'strategy_validation':(280,400),'regime_training':(0,395),'regime_validation':(400,520)},
 {'strategy_training':(120,400),'strategy_validation':(400,520),'regime_training':(120,515),'regime_validation':(520,640)},
 {'strategy_training':(240,520),'strategy_validation':(520,640),'regime_training':(240,635),'regime_validation':(640,760)},
]
OUTER_HELD=(760,1020)


def json_safe(value):
    """Use zero for undefined metrics from an exactly flat/all-cash stream."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return 0.0
    return value

def main():
    offset=260*(OUTER_RUN-1)
    folds=[{k:(v[0]+offset,v[1]+offset) for k,v in fold.items()} for fold in FOLDS]
    outer_held=(OUTER_HELD[0]+offset,OUTER_HELD[1]+offset)
    full_universe=pd.read_parquet(PROJECT/'data'/'processed'/'close_1d_10y.parquet').columns.tolist()
    universe_snapshot=None
    if UNIVERSE_FILTER in {'historical_large','historical_mega'}:
        snapshot_path=ROOT/'work'/'historical_market_cap'/'pretraining_market_caps.parquet'
        snapshots=pd.read_parquet(snapshot_path)
        bucket_column='is_large' if UNIVERSE_FILTER=='historical_large' else 'is_mega'
        chosen=snapshots[(snapshots['outer_run']==OUTER_RUN)&snapshots[bucket_column]].copy()
        universe=[asset for asset in full_universe if asset in set(chosen['asset']) and asset!='SPY']
        if len(universe)<2:
            raise ValueError(f'historical mega-cap universe for outer run {OUTER_RUN} has fewer than two assets')
        method=('point-in-time USD 10 billion <= market cap < USD 200 billion'
                if UNIVERSE_FILTER=='historical_large' else
                'point-in-time market cap >= USD 200 billion')
        universe_snapshot={'method':method,
                           'information_cutoff':str(chosen['information_cutoff'].iloc[0]),
                           'training_start':str(chosen['training_start'].iloc[0]),
                           'assets':universe,'asset_count':len(universe),
                           'market_cap_source':'Yahoo historical shares x split-adjusted close',
                           'benchmark_excluded':['SPY']}
    else:
        universe=full_universe
    prices=get_time_period(universe,time_peri=(0,2060)); returns=prices.pct_change().fillna(0.)
    market=get_time_period(['SPY'],time_peri=(0,2060)).reindex(prices.index)['SPY'].pct_change().fillna(0.)
    volume=pd.read_parquet(PROJECT/'data'/'processed'/'volume_1d_10y.parquet',columns=universe).reindex(prices.index)
    if FEATURE_SET=='autocorrelation_correlation_volatility':
        features=build_hmm_features(market,returns).loc[:,['ac','corr','var']]
        feature_names=['autocorrelation','correlation','volatility']
        feature_slug='ac_corr_vol'
    else:
        features=build_correlation_liquidity_dispersion_features(prices,volume,returns)
        feature_names=['correlation','liquidity','dispersion']
        feature_slug='corr_liq_disp'
    def cfg(name,p): return {'stock_list':universe,'time_period':(0,2060),'freq':'d','strat_class':{name:p},'parameters_':p}
    grids={'cross_asset_mv':[{'z_threshold':z} for z in (1.5,2.,2.5)],
           'momentum_trending':[{'z_threshold':z,'roll':r} for r,z in product((20,30,60),(1.5,2.,2.5))],
           'cross_asset_momentum_trending':[{'z_threshold':z,'roll':r} for r,z in product((20,35,60),(1.5,2.,2.5))]}
    candidate_raw={n:[] for n in SLEEVES}
    for name,grid in grids.items():
        for p in grid:
            c=cfg(name,p); raw=(_get_signals_mv_cross_asset(c) if name=='cross_asset_mv' else
                (_get_signals_momentum_tr(c,prices) if name=='momentum_trending' else _get_signals_momentum_cross_asset(c)))
            candidate_raw[name].append((p,raw.reindex(prices.index).fillna(0.)))
    prepared=[]
    for fold_number,f in enumerate(folds,1):
        periods={'training':f['strategy_training'],'validation':f['strategy_validation']}
        selected={}; sleeves={}
        for name in SLEEVES:
            try:
                selected[name],_=select_strategy_on_validation(name,candidate_raw[name],returns,market,prices.index,periods)
                raw=next(frame for p,frame in candidate_raw[name] if p==selected[name]['parameters'])
                sleeves[name]=normalize(raw,selected[name]['assets'])
            except ValueError as exc:
                if 'no validation candidates' not in str(exc):
                    raise
                selected[name]={'parameters':None,'assets':[],'validation':None,
                                'status':'unavailable: no basket passed the original training eligibility rules'}
                sleeves[name]=pd.DataFrame(0.0,index=prices.index,columns=universe)
        sleeve_net=pd.DataFrame({n:net_returns(frame,returns) for n,frame in sleeves.items()})
        future=pd.DataFrame({n:(1+sleeve_net[n]).rolling(HORIZON).apply(np.prod,raw=True).sub(1).shift(-HORIZON) for n in SLEEVES})
        complete=future.dropna(how='all'); winner=complete.idxmax(axis=1).reindex(future.index); best=complete.max(axis=1).reindex(future.index).fillna(0.)
        labels=pd.DataFrame(0.,index=prices.index,columns=[*SLEEVES,'cash'])
        for n in SLEEVES: labels.loc[(winner==n)&(best>0),n]=1.
        labels.loc[best<=0,'cash']=1.
        prepared.append({'fold':fold_number,'spec':f,'selected_strategies':selected,'sleeves':sleeves,'labels':labels})
        print(f'inner strategy fold {fold_number}/3 complete',flush=True)
    all_columns=list(dict.fromkeys(column for item in prepared for frame in item['sleeves'].values() for column in frame.columns))
    neutral_cache=neutrality_cache(all_columns,returns,market) if NEUTRALITY_MODE!='none' else None
    def target_for(sleeves,allocation):
        target=combine(sleeves,allocation,prices.index)
        return neutralize(target,neutral_cache) if neutral_cache is not None else target
    choices=[]
    transformer_cache={}
    for rebalance in (5,10,20):
        def objective(theta):
            if MODEL_KIND=='decision_tree':
                half,cap,depth,leaf=decode_tree(theta); trees=None
            elif MODEL_KIND=='elastic_net_logistic':
                half,cap,penalty,ratio=decode_elastic(theta); trees=None
            elif MODEL_KIND=='rbf_svm':
                half,cap,c_value,gamma=decode_svm(theta); trees=None
            elif MODEL_KIND=='neural_network':
                half,cap,units,nn_alpha=decode_nn(theta); trees=None
            elif MODEL_KIND=='transformer':
                s=1/(1+np.exp(-np.clip(theta,-20,20))); half=1+19*s[0]; cap=.25+.75*s[1]; architecture=((8,1,.1),(16,1,.2),(16,2,.2))[min(2,int(3*s[2]))]; trees=None
            else:
                half,cap,trees,depth,leaf=decode_forest(theta)
            scores=[]
            for item in prepared:
                f=item['spec']; train=prices.index[slice(*f['regime_training'])]; val=prices.index[slice(*f['regime_validation'])]
                if MODEL_KIND=='decision_tree':
                    model=fit_decision_tree_regime(features.reindex(train),item['labels'].reindex(train),max_depth=depth,min_samples_leaf=leaf)
                    probability=predict_decision_tree_probabilities(model,features.reindex(val))
                elif MODEL_KIND=='elastic_net_logistic':
                    model=fit_elastic_logistic_regime(features.reindex(train),item['labels'].reindex(train),penalty=penalty,l1_ratio=ratio)
                    probability=predict_elastic_probabilities(model,features.reindex(val))
                elif MODEL_KIND=='rbf_svm':
                    model=fit_svm_regime(features.reindex(train),item['labels'].reindex(train),c_values=(c_value,),gamma_values=(gamma,),purge_gap=HORIZON)
                    probability=predict_svm_scores(model,features.reindex(val))
                elif MODEL_KIND=='neural_network':
                    model=fit_nn_regime(features.reindex(train),item['labels'].reindex(train),units,nn_alpha)
                    probability=predict_nn_regime(model,features.reindex(val))
                elif MODEL_KIND=='transformer':
                    key=(item['fold'],architecture)
                    if key not in transformer_cache: transformer_cache[key]=fit_transformer(features.reindex(train),item['labels'].reindex(train),*architecture)
                    context=train[-19:].append(val); probability=predict_transformer(transformer_cache[key],features.reindex(context)).reindex(val).fillna(0.)
                else:
                    model=fit_forest_regime(features.reindex(train),item['labels'].reindex(train),trees,depth,leaf)
                    probability=predict_forest_regime(model,features.reindex(val))
                alloc=allocations(probability,half,rebalance,cap)
                metric=performance(net_returns(target_for(item['sleeves'],alloc),returns).reindex(val).fillna(0.),market.reindex(val))
                scores.append(metric['sharpe'] if np.isfinite(metric['sharpe']) else -1e6)
            return float(np.median(scores))
        dimensions=(3 if MODEL_KIND=='transformer' else (4 if MODEL_KIND in ('decision_tree','elastic_net_logistic','rbf_svm','neural_network') else 5))
        theta=bayesian_optimize(objective,17000+rebalance,dimensions=dimensions)
        if MODEL_KIND=='decision_tree': half,cap,depth,leaf=decode_tree(theta); trees=None
        elif MODEL_KIND=='elastic_net_logistic': half,cap,penalty,ratio=decode_elastic(theta); trees=None
        elif MODEL_KIND=='rbf_svm': half,cap,c_value,gamma=decode_svm(theta); trees=None
        elif MODEL_KIND=='neural_network': half,cap,units,nn_alpha=decode_nn(theta); trees=None
        elif MODEL_KIND=='transformer': s=1/(1+np.exp(-np.clip(theta,-20,20))); half=1+19*s[0]; cap=.25+.75*s[1]; architecture=((8,1,.1),(16,1,.2),(16,2,.2))[min(2,int(3*s[2]))]; trees=None
        else: half,cap,trees,depth,leaf=decode_forest(theta)
        fold_metrics=[]
        for item in prepared:
            f=item['spec']; train=prices.index[slice(*f['regime_training'])]; val=prices.index[slice(*f['regime_validation'])]
            if MODEL_KIND=='decision_tree':
                model=fit_decision_tree_regime(features.reindex(train),item['labels'].reindex(train),max_depth=depth,min_samples_leaf=leaf); probability=predict_decision_tree_probabilities(model,features.reindex(val))
            elif MODEL_KIND=='elastic_net_logistic':
                model=fit_elastic_logistic_regime(features.reindex(train),item['labels'].reindex(train),penalty=penalty,l1_ratio=ratio); probability=predict_elastic_probabilities(model,features.reindex(val))
            elif MODEL_KIND=='rbf_svm':
                model=fit_svm_regime(features.reindex(train),item['labels'].reindex(train),c_values=(c_value,),gamma_values=(gamma,),purge_gap=HORIZON); probability=predict_svm_scores(model,features.reindex(val))
            elif MODEL_KIND=='neural_network':
                model=fit_nn_regime(features.reindex(train),item['labels'].reindex(train),units,nn_alpha); probability=predict_nn_regime(model,features.reindex(val))
            elif MODEL_KIND=='transformer':
                key=(item['fold'],architecture)
                if key not in transformer_cache: transformer_cache[key]=fit_transformer(features.reindex(train),item['labels'].reindex(train),*architecture)
                context=train[-19:].append(val); probability=predict_transformer(transformer_cache[key],features.reindex(context)).reindex(val).fillna(0.)
            else:
                model=fit_forest_regime(features.reindex(train),item['labels'].reindex(train),trees,depth,leaf); probability=predict_forest_regime(model,features.reindex(val))
            alloc=allocations(probability,half,rebalance,cap)
            fold_metrics.append(performance(net_returns(target_for(item['sleeves'],alloc),returns).reindex(val).fillna(0.),market.reindex(val)))
        model_parameters=({'n_estimators':trees,'max_depth':depth,'min_samples_leaf':leaf} if trees is not None else
                          ({'max_depth':depth,'min_samples_leaf':leaf} if MODEL_KIND=='decision_tree' else
                           ({'penalty':penalty,'l1_ratio':ratio} if MODEL_KIND=='elastic_net_logistic' else
                            ({'C':c_value,'gamma':gamma} if MODEL_KIND=='rbf_svm' else
                             ({'hidden_units':units,'alpha':nn_alpha} if MODEL_KIND=='neural_network' else
                              {'d_model':architecture[0],'encoder_layers':architecture[1],'dropout':architecture[2]})))))
        choices.append({**model_parameters,'smoothing_half_life':half,'max_sleeve_weight':cap,'rebalance_every_bars':rebalance,'inner_fold_metrics':fold_metrics,'median_inner_sharpe':float(np.median([m['sharpe'] for m in fold_metrics]))})
        print(f'filter rebalance candidate {rebalance} complete',flush=True)
    choices.sort(key=lambda x:x['median_inner_sharpe'],reverse=True); selected=choices[0]
    final=prepared[-1]; dev=prices.index[offset:offset+760-HORIZON]
    if MODEL_KIND=='decision_tree':
        model=fit_decision_tree_regime(features.reindex(dev),final['labels'].reindex(dev),max_depth=selected['max_depth'],min_samples_leaf=selected['min_samples_leaf']); held=prices.index[slice(*outer_held)]; probability=predict_decision_tree_probabilities(model,features.reindex(held))
    elif MODEL_KIND=='elastic_net_logistic':
        model=fit_elastic_logistic_regime(features.reindex(dev),final['labels'].reindex(dev),penalty=selected['penalty'],l1_ratio=selected['l1_ratio']); held=prices.index[slice(*outer_held)]; probability=predict_elastic_probabilities(model,features.reindex(held))
    elif MODEL_KIND=='rbf_svm':
        model=fit_svm_regime(features.reindex(dev),final['labels'].reindex(dev),c_values=(selected['C'],),gamma_values=(selected['gamma'],),purge_gap=HORIZON); held=prices.index[slice(*outer_held)]; probability=predict_svm_scores(model,features.reindex(held))
    elif MODEL_KIND=='neural_network':
        model=fit_nn_regime(features.reindex(dev),final['labels'].reindex(dev),selected['hidden_units'],selected['alpha']); held=prices.index[slice(*outer_held)]; probability=predict_nn_regime(model,features.reindex(held))
    elif MODEL_KIND=='transformer':
        model=fit_transformer(features.reindex(dev),final['labels'].reindex(dev),selected['d_model'],selected['encoder_layers'],selected['dropout']); held=prices.index[slice(*outer_held)]; context=dev[-19:].append(held); probability=predict_transformer(model,features.reindex(context)).reindex(held).fillna(0.)
    else:
        model=fit_forest_regime(features.reindex(dev),final['labels'].reindex(dev),selected['n_estimators'],selected['max_depth'],selected['min_samples_leaf']); held=prices.index[slice(*outer_held)]; probability=predict_forest_regime(model,features.reindex(held))
    alloc=allocations(probability,selected['smoothing_half_life'],selected['rebalance_every_bars'],selected['max_sleeve_weight'])
    held_target=target_for(final['sleeves'],alloc)
    held_metric=performance(net_returns(held_target,returns).reindex(held).fillna(0.),market.reindex(held))
    model_label=('Decision tree' if MODEL_KIND=='decision_tree' else
                 ('Elastic-net logistic' if MODEL_KIND=='elastic_net_logistic' else
                  ('RBF SVM' if MODEL_KIND=='rbf_svm' else 'Random forest')))
    if MODEL_KIND=='neural_network': model_label='Single-hidden-layer neural network'
    if MODEL_KIND=='transformer': model_label='Causal sequence transformer'
    neutrality_label={'none':[],'beta':['rolling_market_beta'],'pc':['rolling_leading_principal_component']}[NEUTRALITY_MODE]
    output={'test':f'{model_label} outer run {OUTER_RUN} with nested purged chronological validation','outer_run':OUTER_RUN,'features':feature_names,'strategy_set':STRATEGY_SET,'allowed_strategies':list(SLEEVES),'cash_allowed':True,'universe_filter':UNIVERSE_FILTER,'universe_snapshot':universe_snapshot,'neutrality':neutrality_label,'neutrality_lookback_bars':60 if neutral_cache is not None else None,'neutrality_residuals':neutrality_residuals(held_target,neutral_cache,held) if neutral_cache is not None else None,'inner_folds':folds,'purge_bars':HORIZON,'outer_held_out':list(outer_held),'selected_strategy_procedure':'separate strategy validation in each inner fold; final fold selections refit for outer test','undefined_flat_metric_convention':'non-finite metrics from an exactly flat/all-cash return stream are reported as zero','selected_filter':selected,'final_selected_strategies':final['selected_strategies'],'held_out':held_metric,'held_out_passed':passed(held_metric),'execution':{'execution_delay_bars':1,'fee_per_order':FEE,'slippage_per_order':SLIPPAGE},'scientific_status':'Diagnostic: this historical held-out interval was viewed earlier; source universe is the May 2026 constituent file and therefore still has survivorship bias.'}
    output=json_safe(output)
    neutrality_slug={'none':'','beta':'_beta_neutral','pc':'_pc_neutral'}[NEUTRALITY_MODE]
    universe_slug=('_historical_large' if UNIVERSE_FILTER=='historical_large' else
                   ('_historical_mega' if UNIVERSE_FILTER=='historical_mega' else ''))
    strategy_slug='_time_series_momentum' if STRATEGY_SET=='time_series_momentum' else ''
    path=ROOT/'outputs'/f'checkpoint_{MODEL_KIND}_{feature_slug}{universe_slug}{strategy_slug}{neutrality_slug}_nested_purged_outer_run_{OUTER_RUN}_summary.json'; path.write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8'); print(json.dumps(output,indent=2,allow_nan=False))

if __name__=='__main__': main()
