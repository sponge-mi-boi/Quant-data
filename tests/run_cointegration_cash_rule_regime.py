"""Non-ML correlation/liquidity/dispersion filter for cointegration versus cash."""
import json,os,sys
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault('NUMBA_DISABLE_JIT','1')
ROOT  = Path(r"path\to\root"); PROJECT=ROOT/'work'/'PythonProject1_basicbacktester'/'Published'; sys.path[:0]=[str(PROJECT/'src'),str(ROOT/'work')]
from run_cointegration_one_screen_fast import weights_for_pair,FEE,SLIPPAGE,PERIODS
from run_cmv_full_three_stage_five_cycles import performance
from run_cmv_mt_cmt_rule_regime import build_correlation_liquidity_dispersion_features

def allocations(features,half,rebalance,cointegration_scale,cash_scale):
    frame=features.reindex(columns=['corr','liq','dis']).dropna()
    high_corr=((frame['corr']-.5)/.5).clip(0,1); low_corr=((.5-frame['corr'])/.5).clip(0,1)
    high_liq=((frame['liq']-.5)/.5).clip(0,1); low_liq=((.5-frame['liq'])/.5).clip(0,1)
    high_dis=((frame['dis']-.5)/.5).clip(0,1)
    score=pd.DataFrame(index=frame.index)
    score['cointegration']=(.05+low_corr+high_dis+.6*high_liq)*cointegration_scale
    score['cash']=(.05+1.5*low_liq+.5*high_corr)*cash_scale
    weight=score.div(score.sum(axis=1),axis=0).ewm(halflife=half,adjust=False).mean()
    weight.loc[np.arange(len(weight))%rebalance!=0]=np.nan
    return weight.ffill().fillna(0.)

def net_returns(pair_weights,allocation,returns):
    target=pair_weights.mul(allocation['cointegration'],axis=0); executed=target.shift(1).fillna(0.)
    turnover=executed.diff().abs().sum(axis=1).fillna(executed.abs().sum(axis=1))
    return (executed*returns[target.columns]).sum(axis=1)-turnover*(FEE+SLIPPAGE)

def passed(metric): return bool(np.isfinite(metric['sharpe']) and metric['total_return']>0 and metric['sharpe']>0)

def main():
    source=json.loads((ROOT/'outputs'/'cointegration_one_screen_fast_summary.json').read_text()); pair=tuple(source['validation_winner'])
    full=pd.read_parquet(PROJECT/'data'/'processed'/'close_1d_10y.parquet').iloc[:1020]; market=full['SPY'].pct_change().fillna(0.); prices=full.drop(columns='SPY'); returns=prices.pct_change().fillna(0.)
    volume=pd.read_parquet(PROJECT/'data'/'processed'/'volume_1d_10y.parquet',columns=list(prices.columns)).iloc[:1020].reindex(prices.index)
    features=build_correlation_liquidity_dispersion_features(prices,volume,returns); pair_weights=weights_for_pair(pair,prices)
    val=prices.index[slice(*PERIODS['validation'])]; candidates=[]
    for half,rebalance,cointegration_scale,cash_scale in product((2,5,10),(5,10,20),(.5,1,2),(.5,1,2)):
        allocation=allocations(features,half,rebalance,cointegration_scale,cash_scale); metric=performance(net_returns(pair_weights,allocation,returns).reindex(val).fillna(0.),market.reindex(val))
        candidates.append({'smoothing_half_life':half,'rebalance_every_bars':rebalance,'cointegration_scale':cointegration_scale,'cash_scale':cash_scale,'validation':metric})
    candidates.sort(key=lambda x:(passed(x['validation']),x['validation']['sharpe'],x['validation']['total_return']),reverse=True); selected=candidates[0]
    allocation=allocations(features,selected['smoothing_half_life'],selected['rebalance_every_bars'],selected['cointegration_scale'],selected['cash_scale']); held=prices.index[slice(*PERIODS['held_out'])]
    metric=performance(net_returns(pair_weights,allocation,returns).reindex(held).fillna(0.),market.reindex(held)); output={'test':'Cointegration versus cash non-ML rule regime','pair':list(pair),'features':['correlation','liquidity','dispersion'],'classifier':None,'validation_candidates':81,'selected':selected,'average_held_out_allocations':allocation.reindex(held).mean().to_dict(),'held_out':metric,'held_out_passed':passed(metric),'execution':{'delay_bars':1,'fee_per_order':FEE,'slippage_per_order':SLIPPAGE},'scientific_status':'Diagnostic: held-out interval has been viewed previously.'}
    path=ROOT/'outputs'/'cointegration_cash_corr_liq_disp_rule_regime_summary.json'; path.write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8'); print(json.dumps(output,indent=2,allow_nan=False))
if __name__=='__main__': main()
