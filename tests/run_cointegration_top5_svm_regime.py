"""Top-five parameter-optimized cointegration gated by an RBF-SVM regime model."""
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault('NUMBA_DISABLE_JIT','1')
ROOT  = Path(r"path\to\root")

PROJECT=ROOT/'work'/'PythonProject1_basicbacktester'/'Published'
sys.path[:0]=[str(PROJECT/'src'),str(ROOT/'work')]

from src import _get_signals
from src import fit_svm_regime,predict_svm_scores
from run_cmv_full_three_stage_five_cycles import CYCLES,FEE,SLIPPAGE,performance
from run_cmv_mt_cmt_rule_regime import build_correlation_liquidity_dispersion_features
from run_cmv_mt_cmt_logistic_regime import bayesian_optimize,decode_svm

HORIZON=5
REBALANCES=(5,10,20)

def net_from_target(target,returns):
    executed=target.shift(1).fillna(0.)
    turnover=executed.diff().abs().sum(axis=1).fillna(executed.abs().sum(axis=1))
    return (executed*returns[target.columns]).sum(axis=1)-turnover*(FEE+SLIPPAGE)

def exposure(scores,half,rebalance,cap):
    value=scores['cointegration'].clip(0.,cap).ewm(halflife=half,adjust=False).mean()
    value.loc[np.arange(len(value))%rebalance!=0]=np.nan
    return value.ffill().fillna(0.)

def passed(metric):
    return bool(np.isfinite(metric['sharpe']) and metric['sharpe']>0 and metric['total_return']>0)

def main():
    selection=json.loads((ROOT/'outputs'/'checkpoint_pure_cointegration_top_5_pairs_parameter_optimized_five_cycles_summary.json').read_text())
    all_prices=pd.read_parquet(PROJECT/'data'/'processed'/'close_1d_10y.parquet').iloc[:2060]
    market=all_prices['SPY'].pct_change().fillna(0.); prices=all_prices.drop(columns='SPY')
    returns=prices.pct_change().fillna(0.)
    volume=pd.read_parquet(PROJECT/'data'/'processed'/'volume_1d_10y.parquet',columns=list(prices.columns)).reindex(prices.index)
    features=build_correlation_liquidity_dispersion_features(prices,volume,returns)
    cache={}; runs=[]
    def pair_target(member):
        pair=tuple(member['pair']); p=member['parameters']
        threshold=tuple(p['z_threshold']) if isinstance(p['z_threshold'],list) else p['z_threshold']
        key=(pair,threshold,p['roll'])
        if key not in cache:
            cfg={'stock_list':list(pair),'parameters_':{'z_threshold':threshold,'roll':p['roll']},'weights_filter':{}}
            raw=_get_signals(cfg,prices[list(pair)]).reindex(prices.index).fillna(0.)
            cache[key]=raw.div(raw.abs().sum(axis=1).replace(0,np.nan),axis=0).fillna(0.)
        return cache[key]
    for number,(periods,prior) in enumerate(zip(CYCLES,selection['runs']),1):
        members=prior['selected']['members']; columns=list(dict.fromkeys(a for m in members for a in m['pair']))
        base=pd.DataFrame(0.,index=prices.index,columns=columns)
        for member in members:
            target=pair_target(member); base.loc[:,target.columns]+=target/len(members)
        base_net=net_from_target(base,returns)
        forward=(1+base_net).rolling(HORIZON).apply(np.prod,raw=True).sub(1).shift(-HORIZON)
        labels=pd.DataFrame({'cointegration':(forward>0).astype(float),'cash':(forward<=0).astype(float)},index=prices.index)
        train=prices.index[periods['training'][0]:periods['training'][1]-HORIZON]
        val=prices.index[slice(*periods['validation'])]; candidates=[]
        for rebalance in REBALANCES:
            def objective(theta):
                half,cap,c,gamma=decode_svm(theta)
                model=fit_svm_regime(features.reindex(train),labels.reindex(train),c_values=(c,),gamma_values=(gamma,),purge_gap=HORIZON)
                scores=predict_svm_scores(model,features.reindex(val)); exp=exposure(scores,half,rebalance,cap)
                metric=performance(net_from_target(base.mul(exp,axis=0),returns).reindex(val).fillna(0.),market.reindex(val))
                return float(metric['sharpe']) if np.isfinite(metric['sharpe']) else -1e6
            theta=bayesian_optimize(objective,9000+number*100+rebalance,dimensions=4)
            half,cap,c,gamma=decode_svm(theta)
            model=fit_svm_regime(features.reindex(train),labels.reindex(train),c_values=(c,),gamma_values=(gamma,),purge_gap=HORIZON)
            scores=predict_svm_scores(model,features.reindex(val)); exp=exposure(scores,half,rebalance,cap)
            metric=performance(net_from_target(base.mul(exp,axis=0),returns).reindex(val).fillna(0.),market.reindex(val))
            candidates.append({'C':c,'gamma':gamma,'smoothing_half_life':half,'max_weight':cap,'rebalance_every_bars':rebalance,'validation':metric})
        candidates.sort(key=lambda x:(passed(x['validation']),x['validation']['sharpe'],x['validation']['total_return']),reverse=True)
        selected=candidates[0]
        dev=prices.index[periods['training'][0]:periods['validation'][1]-HORIZON]
        model=fit_svm_regime(features.reindex(dev),labels.reindex(dev),c_values=(selected['C'],),gamma_values=(selected['gamma'],),purge_gap=HORIZON)
        held=prices.index[slice(*periods['held_out'])]; scores=predict_svm_scores(model,features.reindex(held))
        exp=exposure(scores,selected['smoothing_half_life'],selected['rebalance_every_bars'],selected['max_weight'])
        metric=performance(net_from_target(base.mul(exp,axis=0),returns).reindex(held).fillna(0.),market.reindex(held))
        runs.append({'run':number,'pairs':members,'selected_regime':selected,'average_held_out_cointegration_weight':float(exp.reindex(held).mean()),'held_out':metric,'held_out_passed':passed(metric)})
        print(f'cycle {number}/5 complete',flush=True)
    names=('total_return','sharpe','alpha','max_drawdown')
    output={'test':'Top-five shared-threshold cointegration with RBF-SVM regime vs cash','features':['correlation','liquidity','dispersion'],'classifier':'gaussian_rbf_svm_bayesian_optimized','cointegration_parameters':{'entry_z_optimized':[1.5,2.0,2.5],'exit_z_fixed':0.0,'roll_optimized':[20,32,60]},'allowed_sleeves':['cointegration','cash'],'target_horizon_bars':HORIZON,'purge_bars':HORIZON,'regime_filter':True,'execution':{'execution_delay_bars':1,'fee_per_order':FEE,'slippage_per_order':SLIPPAGE},'runs':runs,'average_held_out_metrics':{n:float(np.mean([r['held_out'][n] for r in runs])) for n in names},'held_out_pass_count':sum(r['held_out_passed'] for r in runs),'scientific_status':'Diagnostic: these historical held-out windows were viewed earlier.'}
    path=ROOT/'outputs'/'checkpoint_pure_cointegration_top5_shared_threshold_corr_liq_disp_rbf_svm_regime_summary.json'
    path.write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8'); print(json.dumps(output,indent=2,allow_nan=False))

if __name__=='__main__': main()
