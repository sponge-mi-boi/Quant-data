import json, os, sys
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault('NUMBA_DISABLE_JIT', '1')
ROOT = Path(r"path\to\root")
PROJECT = ROOT / 'work' / 'PythonProject1_basicbacktester' / 'Published'
sys.path[:0] = [str(PROJECT / 'src'), str(ROOT / 'work')]

from src.data_filter import get_time_period
from src.hmm_regime import build_variance_dispersion_trend_features
from src.strategies import _get_signals_mv_cross_asset, _get_signals_momentum_tr
from src.tree_regime import fit_tree_regime, predict_tree_probabilities
from src.svm_regime import fit_svm_regime, predict_svm_scores
from src.logistic_regime import fit_logistic_regime, predict_regime_probabilities
from src.elastic_logistic_regime import (
    fit_elastic_logistic_regime, predict_elastic_probabilities)
from run_cmv_full_three_stage_five_cycles import CYCLES, FEE, SLIPPAGE, performance

HORIZON = 5
STRATEGY_SET = os.environ.get('CMV_MOMENTUM_STRATEGIES', 'both')
SLEEVES = (('momentum_trending',) if STRATEGY_SET == 'time_series_momentum'
           else ('cross_asset_mv', 'momentum_trending'))
MODEL_KIND = os.environ.get('CMV_MOMENTUM_MODEL', 'random_forest')


def normalized(raw, assets):
    frame = raw[assets].copy()
    return frame.div(frame.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0)


def combine(cmv, momentum, allocation):
    columns = list(dict.fromkeys(list(cmv.columns) + list(momentum.columns)))
    target = pd.DataFrame(0., index=cmv.index, columns=columns)
    if 'cross_asset_mv' in allocation:
        target.loc[:, cmv.columns] += cmv.mul(allocation['cross_asset_mv'], axis=0)
    if 'momentum_trending' in allocation:
        target.loc[:, momentum.columns] += momentum.mul(allocation['momentum_trending'], axis=0)
    return target


def net_returns(weights, returns):
    executed = weights.shift(1).fillna(0)
    turnover = executed.diff().abs().sum(axis=1).fillna(executed.abs().sum(axis=1))
    return (executed * returns[weights.columns]).sum(axis=1) - turnover * (FEE + SLIPPAGE)


def allocations(probabilities, half_life, rebalance, cap):
    weights = probabilities[list(SLEEVES)].clip(upper=cap)
    weights = weights.ewm(halflife=half_life, adjust=False).mean()
    weights.loc[np.arange(len(weights)) % rebalance != 0] = np.nan
    return weights.ffill().fillna(0)


def passed(metrics):
    return bool(np.isfinite(metrics['sharpe']) and metrics['total_return'] > 0 and metrics['sharpe'] > 0)


def main():
    cmv_source = json.loads((ROOT/'outputs'/'checkpoint_cross_asset_mv_nonneutral_three_stage_five_cycles_summary.json').read_text())
    mom_source = json.loads((ROOT/'outputs'/'checkpoint_momentum_trending_nonneutral_three_stage_five_cycles_summary.json').read_text())
    universe = pd.read_parquet(PROJECT/'data'/'processed'/'close_1d_10y.parquet').columns.tolist()
    prices = get_time_period(universe, time_peri=(0, 2060)); returns = prices.pct_change().fillna(0)
    market = get_time_period(['SPY'], time_peri=(0,2060)).reindex(prices.index)['SPY'].pct_change().fillna(0)
    cmv_params = {'stock_list': universe, 'time_period': (0,2060), 'freq':'d',
        'strat_class': {'cross_asset_mv': {'z_threshold':2.}}, 'parameters_': {'z_threshold':2.}}
    mom_params = {'stock_list': universe, 'time_period': (0,2060), 'freq':'d',
        'strat_class': {'momentum_trending': {'z_threshold':1.999,'roll':30}},
        'parameters_': {'z_threshold':1.999,'roll':30}}
    cmv_raw = _get_signals_mv_cross_asset(cmv_params).reindex(prices.index).fillna(0)
    mom_raw = _get_signals_momentum_tr(mom_params, prices).reindex(prices.index).fillna(0)
    features = build_variance_dispersion_trend_features(market, returns)
    runs=[]
    for number, (periods, cr, mr) in enumerate(zip(CYCLES, cmv_source['runs'], mom_source['runs']),1):
        cmv = normalized(cmv_raw, cr['validation_winner']); mom = normalized(mom_raw, mr['validation_winner'])
        all_sleeve_net = {'cross_asset_mv': net_returns(cmv,returns),
                          'momentum_trending': net_returns(mom,returns)}
        sleeve_net = pd.DataFrame({name: all_sleeve_net[name] for name in SLEEVES})
        future = pd.DataFrame({c:(1+sleeve_net[c]).rolling(HORIZON).apply(np.prod,raw=True).sub(1).shift(-HORIZON)
                               for c in SLEEVES})
        complete_future = future.dropna(how='all')
        winner = complete_future.idxmax(axis=1).reindex(future.index)
        best = complete_future.max(axis=1).reindex(future.index).fillna(0.0)
        labels = pd.DataFrame(0.,index=prices.index,columns=[*SLEEVES,'cash'])
        for name in SLEEVES: labels.loc[(winner==name)&(best>0),name]=1
        labels.loc[best<=0,'cash']=1
        train=prices.index[periods['training'][0]:periods['training'][1]-HORIZON]; val=prices.index[slice(*periods['validation'])]
        candidates=[]
        model_grid = (list(product((2,4,8),(10,30,60))) if MODEL_KIND == 'random_forest' else
                      (list(product((.001,.01,.1,1.,10.),(.25,.5,.75))) if MODEL_KIND == 'elastic' else
                      ([(p,None) for p in (.001,.01,.1,1.,10.)] if MODEL_KIND in ('lasso','ridge') else
                      ([(0.,None)] if MODEL_KIND == 'logistic_none'
                       else list(product((.1,1.,10.),('scale',.1,1.)))))))
        for first,second in model_grid:
            if MODEL_KIND == 'random_forest':
                model=fit_tree_regime(features.reindex(train),labels.reindex(train),max_depth=first,min_samples_leaf=second)
                probability=predict_tree_probabilities(model,features.reindex(val))
            elif MODEL_KIND == 'rbf_svm':
                model=fit_svm_regime(features.reindex(train),labels.reindex(train),c_values=(first,),gamma_values=(second,),purge_gap=HORIZON)
                probability=predict_svm_scores(model,features.reindex(val))
            elif MODEL_KIND in ('lasso','elastic'):
                model=fit_elastic_logistic_regime(features.reindex(train),labels.reindex(train),penalty=first,
                                                   l1_ratio=(second if MODEL_KIND == 'elastic' else 1.))
                probability=predict_elastic_probabilities(model,features.reindex(val))
            else:
                model=fit_logistic_regime(features.reindex(train),labels.reindex(train),
                                          l2=(first if MODEL_KIND == 'ridge' else 0.))
                probability=predict_regime_probabilities(model,features.reindex(val))
            for half,rebalance,cap in product((2,5,10),(5,10,20),(.5,1.)):
                allocation=allocations(probability,half,rebalance,cap)
                net=net_returns(combine(cmv,mom,allocation),returns).reindex(val).fillna(0)
                if MODEL_KIND == 'random_forest':
                    model_params={'max_depth':first,'min_samples_leaf':second}
                elif MODEL_KIND == 'rbf_svm':
                    model_params={'C':first,'gamma':second}
                elif MODEL_KIND == 'elastic':
                    model_params={'penalty':first,'l1_ratio':second}
                elif MODEL_KIND == 'lasso':
                    model_params={'l1_penalty':first}
                elif MODEL_KIND == 'ridge':
                    model_params={'l2_penalty':first}
                else:
                    model_params={'l2':0.}
                metric=performance(net,market.reindex(val)); candidates.append({**model_params,
                    'smoothing_half_life':half,'rebalance_every_bars':rebalance,'max_sleeve_weight':cap,'validation':metric})
        candidates.sort(key=lambda x:(passed(x['validation']),x['validation']['sharpe'],x['validation']['total_return']),reverse=True)
        selected=candidates[0]; dev=prices.index[periods['training'][0]:periods['validation'][1]-HORIZON]
        if MODEL_KIND == 'random_forest':
            model=fit_tree_regime(features.reindex(dev),labels.reindex(dev),max_depth=selected['max_depth'],min_samples_leaf=selected['min_samples_leaf'])
        elif MODEL_KIND == 'rbf_svm':
            model=fit_svm_regime(features.reindex(dev),labels.reindex(dev),c_values=(selected['C'],),gamma_values=(selected['gamma'],),purge_gap=HORIZON)
        elif MODEL_KIND in ('lasso','elastic'):
            model=fit_elastic_logistic_regime(features.reindex(dev),labels.reindex(dev),
                                               penalty=(selected['penalty'] if MODEL_KIND == 'elastic' else selected['l1_penalty']),
                                               l1_ratio=(selected['l1_ratio'] if MODEL_KIND == 'elastic' else 1.))
        else:
            model=fit_logistic_regime(features.reindex(dev),labels.reindex(dev),
                l2=(selected['l2_penalty'] if MODEL_KIND == 'ridge' else 0.))
        held=prices.index[slice(*periods['held_out'])]
        probability=(predict_tree_probabilities(model,features.reindex(held)) if MODEL_KIND == 'random_forest' else
                     (predict_svm_scores(model,features.reindex(held)) if MODEL_KIND == 'rbf_svm' else
                      (predict_elastic_probabilities(model,features.reindex(held)) if MODEL_KIND in ('lasso','elastic')
                       else predict_regime_probabilities(model,features.reindex(held)))))
        allocation=allocations(probability,selected['smoothing_half_life'],selected['rebalance_every_bars'],selected['max_sleeve_weight'])
        metric=performance(net_returns(combine(cmv,mom,allocation),returns).reindex(held).fillna(0),market.reindex(held))
        runs.append({'run':number,'selected':selected,'held_out':metric,'held_out_passed':passed(metric)})
        print(f'cycle {number}/5 complete',flush=True)
    names=('total_return','sharpe','alpha','max_drawdown')
    classifier={'random_forest':'random_forest','rbf_svm':'gaussian_rbf_svm',
                'elastic':'elastic_net_logistic','lasso':'lasso_logistic',
                'ridge':'ridge_logistic','logistic_none':'unregularized_logistic'}[MODEL_KIND]
    strategy_label=('time-series momentum + cash' if STRATEGY_SET == 'time_series_momentum'
                    else 'CMV + time-series momentum + cash')
    output={'test':f'{strategy_label} {classifier}','features':['variance','dispersion','trend'],
        'classifier':classifier,'strategies':list(SLEEVES),'runs':runs,
        'average_held_out_metrics':{n:float(np.mean([r['held_out'][n] for r in runs])) for n in names},
        'held_out_pass_count':sum(r['held_out_passed'] for r in runs),'scientific_status':'Diagnostic: held-out windows were viewed earlier.'}
    prefix=('momentum_only' if STRATEGY_SET == 'time_series_momentum' else 'cmv_momentum')
    path=ROOT/'outputs'/f'checkpoint_{prefix}_three_feature_{MODEL_KIND}_summary.json'
    path.write_text(json.dumps(output,indent=2,allow_nan=False)); print(json.dumps(output,indent=2,allow_nan=False))

if __name__=='__main__': main()
