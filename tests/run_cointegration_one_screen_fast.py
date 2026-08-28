"""Fast single-cycle cointegration screen with continuous causal warm-up."""
import json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault('NUMBA_DISABLE_JIT','1')
ROOT = Path (__file__).resolve().parent.parent; PROJECT = ROOT;sys.path.insert(0,str(PROJECT/'src/quant_backtester'))
from src.quant_backtester .strategies import _get_signals
from run_cmv_full_three_stage_five_cycles import performance
from run_cointegration_all_pairs import benjamini_hochberg

PERIODS={'training':(0,500),'validation':(500,760),'held_out':(760,1020)}
FEE=.0005; SLIPPAGE=.0005; PARAMETERS={'z_threshold':1.92,'roll':32}

def weights_for_pair(pair,prices):
    params={'stock_list':list(pair),'parameters_':PARAMETERS,'weights_filter':{}}
    raw=_get_signals(params,prices[list(pair)]).reindex(prices.index).fillna(0.)
    return raw.div(raw.abs().sum(axis=1).replace(0,np.nan),axis=0).fillna(0.)

def evaluate(pair,prices,returns,market,period):
    weights=weights_for_pair(pair,prices); executed=weights.shift(1).fillna(0.)
    turnover=executed.diff().abs().sum(axis=1).fillna(executed.abs().sum(axis=1))
    net=(executed*returns[list(pair)]).sum(axis=1)-turnover*(FEE+SLIPPAGE)
    index=prices.index[slice(*period)]; metric=performance(net.reindex(index).fillna(0.),market.reindex(index))
    metric['position_changes']=int(executed.reindex(index).diff().abs().sum(axis=1).gt(1e-12).sum())
    return metric

def main():
    prices=pd.read_parquet(PROJECT/'data'/'close_1d_10y.parquet').iloc[:1020]
    market=prices['SPY'].pct_change().fillna(0.); assets=prices.drop(columns='SPY'); returns=assets.pct_change().fillna(0.)
    screen=pd.read_csv(ROOT/'artifacts'/'checkpoint_costs_cointegration_test1_all_pair_pvalues.csv'); screen['q_value']=benjamini_hochberg(screen.p_value.to_numpy()); survivors=screen[screen.q_value<=.05].sort_values('q_value')
    training=[]
    for row in survivors[['asset_1','asset_2']].itertuples(index=False,name=None):
        metric=evaluate(row,assets,returns,market,PERIODS['training'])
        if np.isfinite(metric['sharpe']) and metric['position_changes']>=5: training.append((row,metric))
    training.sort(key=lambda x:(x[1]['sharpe'],x[1]['total_return']),reverse=True); top=training[:10]
    validation=[]
    for pair,_ in top:
        metric=evaluate(pair,assets,returns,market,PERIODS['validation'])
        if np.isfinite(metric['sharpe']) and metric['position_changes']>=3: validation.append((pair,metric))
    validation.sort(key=lambda x:(x[1]['sharpe'],x[1]['total_return']),reverse=True)
    if not validation: raise ValueError('No validation-eligible pair')
    winner,validation_metric=validation[0]; held=evaluate(winner,assets,returns,market,PERIODS['held_out'])
    output={'test':'One fast all-pairs cointegration screen','pairs_screened':int(len(screen)),'fdr_survivors':int(len(survivors)),'training_eligible':len(training),'training_top_10':[{'pair':list(p),'metrics':m} for p,m in top],'validation_results':[{'pair':list(p),'metrics':m} for p,m in validation],'validation_winner':list(winner),'validation':validation_metric,'held_out':held,'held_out_passed':bool(held['total_return']>0 and held['sharpe']>0),'parameters':PARAMETERS,'execution':{'delay_bars':1,'fee_per_order':FEE,'slippage_per_order':SLIPPAGE,'continuous_preperiod_warmup':True},'screening':{'engle_granger_orientation':'asset_1_on_asset_2','fdr_alpha':.05},'scientific_status':'Diagnostic: held-out interval has been viewed in prior experiments.'}
    path=ROOT/'artifacts'/'cointegration_one_screen_fast_summary.json'; path.write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8'); print(json.dumps(output,indent=2,allow_nan=False))
if __name__=='__main__': main()
