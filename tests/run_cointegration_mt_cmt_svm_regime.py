"""RBF-SVM allocator: cointegration, time-series momentum, cross momentum, cash."""
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault('NUMBA_DISABLE_JIT','1')
ROOT  = Path(r"path\to\root")
PROJECT=ROOT/'work'/'PythonProject1_basicbacktester'/'Published'
sys.path[:0]=[str(PROJECT/'src'),str(ROOT/'work')]
from src import (_get_signals, _get_signals_mv_cross_asset,
                 _get_signals_momentum_tr, _get_signals_momentum_cross_asset)
from src import fit_svm_regime,predict_svm_scores
from run_cmv_full_three_stage_five_cycles import CYCLES,FEE,SLIPPAGE,performance
from run_cross_momentum_timeseries_momentum_rule_regime import normalize,net_returns,passed
from run_cmv_mt_cmt_rule_regime import build_correlation_liquidity_dispersion_features
from run_cmv_mt_cmt_logistic_regime import bayesian_optimize,decode_svm

HORIZON=5; REBALANCES=(5,10,20)
SLEEVES=('cointegration','cross_asset_mv','momentum_trending','cross_asset_momentum_trending')

def allocations(probability,half,rebalance,cap,cointegration_hurdle):
    gated=probability[list(SLEEVES)].copy()
    gated['cointegration']=gated['cointegration'].where(
        gated['cointegration']>=cointegration_hurdle,0.)
    weight=gated.clip(upper=cap).ewm(halflife=half,adjust=False).mean()
    weight.loc[np.arange(len(weight))%rebalance!=0]=np.nan
    return weight.ffill().fillna(0.)

def decode_with_hurdle(theta):
    half,cap,c,gamma=decode_svm(theta[:4])
    hurdle=.50+.40/(1+np.exp(-np.clip(theta[4],-20,20)))
    return half,cap,c,gamma,hurdle

def combine(sleeves,allocation,index):
    columns=list(dict.fromkeys(c for frame in sleeves.values() for c in frame.columns))
    target=pd.DataFrame(0.,index=index,columns=columns)
    for name,frame in sleeves.items(): target.loc[:,frame.columns]+=frame.mul(allocation[name],axis=0)
    return target

def main():
    coin_source=json.loads((ROOT/'outputs'/'checkpoint_pure_cointegration_top_5_pairs_parameter_optimized_five_cycles_summary.json').read_text())
    best_source=json.loads((ROOT/'outputs'/'checkpoint_three_strategy_bayesian_optimized_corr_liq_disp_rbf_svm_regime_summary.json').read_text())
    all_prices=pd.read_parquet(PROJECT/'data'/'processed'/'close_1d_10y.parquet').iloc[:2060]
    market=all_prices['SPY'].pct_change().fillna(0.); prices=all_prices.drop(columns='SPY'); returns=prices.pct_change().fillna(0.)
    universe=list(prices.columns); volume=pd.read_parquet(PROJECT/'data'/'processed'/'volume_1d_10y.parquet',columns=universe).reindex(prices.index)
    features=build_correlation_liquidity_dispersion_features(prices,volume,returns); cache={}; runs=[]
    def cfg(name,p): return {'stock_list':universe,'time_period':(0,2060),'freq':'d','strat_class':{name:p},'parameters_':p}
    def pair_target(member):
        pair=tuple(member['pair']); p=member['parameters']; key=(pair,p['z_threshold'],p['roll'])
        if key not in cache:
            raw=_get_signals({'stock_list':list(pair),'parameters_':p,'weights_filter':{}},prices[list(pair)]).reindex(prices.index).fillna(0.)
            cache[key]=raw.div(raw.abs().sum(axis=1).replace(0,np.nan),axis=0).fillna(0.)
        return cache[key]
    for number,(periods,coin_prior,best_prior) in enumerate(zip(CYCLES,coin_source['runs'],best_source['runs']),1):
        members=coin_prior['selected']['members']; coin_cols=list(dict.fromkeys(a for m in members for a in m['pair']))
        coin=pd.DataFrame(0.,index=prices.index,columns=coin_cols)
        for member in members:
            target=pair_target(member); coin.loc[:,target.columns]+=target/len(members)
        chosen=best_prior['selected_strategies']; sleeves={'cointegration':coin}
        cmv=chosen['cross_asset_mv']; mt=chosen['momentum_trending']; cmt=chosen['cross_asset_momentum_trending']
        sleeves['cross_asset_mv']=normalize(_get_signals_mv_cross_asset(cfg('cross_asset_mv',cmv['parameters'])).reindex(prices.index).fillna(0.),cmv['assets'])
        sleeves['momentum_trending']=normalize(_get_signals_momentum_tr(cfg('momentum_trending',mt['parameters']),prices).reindex(prices.index).fillna(0.),mt['assets'])
        sleeves['cross_asset_momentum_trending']=normalize(_get_signals_momentum_cross_asset(cfg('cross_asset_momentum_trending',cmt['parameters'])).reindex(prices.index).fillna(0.),cmt['assets'])
        sleeve_net=pd.DataFrame({n:net_returns(f,returns) for n,f in sleeves.items()})
        future=pd.DataFrame({n:(1+sleeve_net[n]).rolling(HORIZON).apply(np.prod,raw=True).sub(1).shift(-HORIZON) for n in SLEEVES})
        complete=future.dropna(how='all'); winner=complete.idxmax(axis=1).reindex(future.index); best=complete.max(axis=1).reindex(future.index).fillna(0.)
        labels=pd.DataFrame(0.,index=prices.index,columns=[*SLEEVES,'cash'])
        for n in SLEEVES: labels.loc[(winner==n)&(best>0),n]=1.
        labels.loc[best<=0,'cash']=1.
        train=prices.index[periods['training'][0]:periods['training'][1]-HORIZON]; val=prices.index[slice(*periods['validation'])]; candidates=[]
        for rebalance in REBALANCES:
            def objective(theta):
                half,cap,c,gamma,hurdle=decode_with_hurdle(theta); model=fit_svm_regime(features.reindex(train),labels.reindex(train),c_values=(c,),gamma_values=(gamma,),purge_gap=HORIZON)
                prob=predict_svm_scores(model,features.reindex(val)); alloc=allocations(prob,half,rebalance,cap,hurdle)
                metric=performance(net_returns(combine(sleeves,alloc,prices.index),returns).reindex(val).fillna(0.),market.reindex(val))
                return float(metric['sharpe']) if np.isfinite(metric['sharpe']) else -1e6
            theta=bayesian_optimize(objective,12000+number*100+rebalance,dimensions=5); half,cap,c,gamma,hurdle=decode_with_hurdle(theta)
            model=fit_svm_regime(features.reindex(train),labels.reindex(train),c_values=(c,),gamma_values=(gamma,),purge_gap=HORIZON)
            prob=predict_svm_scores(model,features.reindex(val)); alloc=allocations(prob,half,rebalance,cap,hurdle)
            metric=performance(net_returns(combine(sleeves,alloc,prices.index),returns).reindex(val).fillna(0.),market.reindex(val))
            candidates.append({'C':c,'gamma':gamma,'cointegration_confidence_hurdle':hurdle,'smoothing_half_life':half,'max_sleeve_weight':cap,'rebalance_every_bars':rebalance,'validation':metric})
        candidates.sort(key=lambda x:(passed(x['validation']),x['validation']['sharpe'],x['validation']['total_return']),reverse=True); selected=candidates[0]
        dev=prices.index[periods['training'][0]:periods['validation'][1]-HORIZON]
        model=fit_svm_regime(features.reindex(dev),labels.reindex(dev),c_values=(selected['C'],),gamma_values=(selected['gamma'],),purge_gap=HORIZON)
        held=prices.index[slice(*periods['held_out'])]; prob=predict_svm_scores(model,features.reindex(held)); alloc=allocations(prob,selected['smoothing_half_life'],selected['rebalance_every_bars'],selected['max_sleeve_weight'],selected['cointegration_confidence_hurdle'])
        metric=performance(net_returns(combine(sleeves,alloc,prices.index),returns).reindex(held).fillna(0.),market.reindex(held)); average=alloc.reindex(held).mean().to_dict(); average['cash']=float(1-alloc.reindex(held).sum(axis=1).mean())
        runs.append({'run':number,'cointegration_pairs':members,'other_strategy_selections':{'cross_asset_mv':cmv,'momentum_trending':mt,'cross_asset_momentum_trending':cmt},'selected_regime':selected,'average_held_out_allocations':average,'held_out':metric,'held_out_passed':passed(metric)}); print(f'cycle {number}/5 complete',flush=True)
    names=('total_return','sharpe','alpha','max_drawdown')
    output={'test':'RBF-SVM regime: cointegration confidence hurdle + CMV + MT + CMT + cash','features':['correlation','liquidity','dispersion'],'classifier':'gaussian_rbf_svm_bayesian_optimized','allowed_sleeves':[*SLEEVES,'cash'],'cointegration_confidence_hurdle_range':[.5,.9],'cointegration_parameters':{'entry_z_optimized':[1.5,2.,2.5],'exit_z_fixed':0.,'roll_optimized':[20,32,60],'pairs_per_run':5},'target_horizon_bars':HORIZON,'purge_bars':HORIZON,'execution':{'execution_delay_bars':1,'fee_per_order':FEE,'slippage_per_order':SLIPPAGE},'runs':runs,'average_held_out_metrics':{n:float(np.mean([r['held_out'][n] for r in runs])) for n in names},'held_out_pass_count':sum(r['held_out_passed'] for r in runs),'scientific_status':'Diagnostic: these historical held-out windows were viewed earlier.'}
    path=ROOT/'outputs'/'checkpoint_cointegration_hurdle_cmv_mt_cmt_corr_liq_disp_rbf_svm_regime_summary.json'; path.write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8'); print(json.dumps(output,indent=2,allow_nan=False))

if __name__=='__main__': main()
