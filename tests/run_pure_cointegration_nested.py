"""Pure singleton-pair cointegration with nested chronological validation."""
import json, os, sys
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

os.environ.setdefault('NUMBA_DISABLE_JIT','1')
ROOT  = Path(r"path\to\root")
PROJECT=ROOT/'work'/'PythonProject1_basicbacktester'/'Published'
sys.path[:0]=[str(PROJECT/'src'),str(ROOT/'work')]
from src.strategies import _get_signals
from run_cmv_full_three_stage_five_cycles import FEE,SLIPPAGE,performance
from run_cointegration_all_pairs import benjamini_hochberg

BASE_FOLDS=[
    {'training':(0,280),'validation':(280,400)},
    {'training':(120,400),'validation':(400,520)},
    {'training':(240,520),'validation':(520,640)},
]
OUTER_HELD=(760,1020)
PARAMETERS=[{'roll':roll,'z_threshold':(entry,exit_)}
            for roll in (20,32,60) for entry in (1.5,2.,2.5)
            for exit_ in (0.,.5,1.) if exit_<entry]
SHORTLIST=100

def main():
    all_prices=pd.read_parquet(PROJECT/'data'/'processed'/'close_1d_10y.parquet').iloc[:2060]
    market=all_prices['SPY'].pct_change().fillna(0.)
    prices=all_prices.drop(columns='SPY'); returns=prices.pct_change().fillna(0.)
    screen=pd.read_csv(ROOT/'work'/'cointegration_all_pairs_screen.csv')
    screen['q_value']=benjamini_hochberg(screen.p_value.to_numpy())
    pairs=[tuple(row) for row in screen.loc[screen.q_value<=.05,['asset_1','asset_2']].itertuples(index=False,name=None)]
    cache={}
    def series(pair,params):
        key=(pair,params['roll'],tuple(params['z_threshold']))
        if key not in cache:
            raw=_get_signals({'stock_list':list(pair),'parameters_':params,'weights_filter':{}},prices[list(pair)]).reindex(prices.index).fillna(0.)
            target=raw.div(raw.abs().sum(axis=1).replace(0,np.nan),axis=0).fillna(0.)
            executed=target.shift(1).fillna(0.); turnover=executed.diff().abs().sum(axis=1).fillna(executed.abs().sum(axis=1))
            net=(executed*returns[list(pair)]).sum(axis=1)-turnover*(FEE+SLIPPAGE)
            changes=executed.diff().abs().sum(axis=1).gt(1e-12)
            cache[key]=(net,changes)
        return cache[key]
    def evaluate(pair,params,index):
        net,changes=series(pair,params); result=performance(net.reindex(index).fillna(0.),market.reindex(index))
        result['position_changes']=int(changes.reindex(index).fillna(False).sum())
        return result
    runs=[]
    for outer_run in range(1,6):
        shift=260*(outer_run-1)
        folds=[{name:(a+shift,b+shift) for name,(a,b) in fold.items()} for fold in BASE_FOLDS]
        held_range=(OUTER_HELD[0]+shift,OUTER_HELD[1]+shift)
        training_rank=[]
        for pair in pairs:
            for params in PARAMETERS:
                metrics=[evaluate(pair,params,prices.index[slice(*fold['training'])]) for fold in folds]
                if all(np.isfinite(m['sharpe']) and m['position_changes']>=3 for m in metrics):
                    training_rank.append((median(m['sharpe'] for m in metrics),pair,params,metrics))
        training_rank.sort(key=lambda row:row[0],reverse=True)
        validation=[]
        for _,pair,params,training_metrics in training_rank[:SHORTLIST]:
            metrics=[evaluate(pair,params,prices.index[slice(*fold['validation'])]) for fold in folds]
            finite=[m['sharpe'] for m in metrics if np.isfinite(m['sharpe']) and m['position_changes']>=2]
            if len(finite)==3:
                validation.append({'pair':list(pair),'parameters':params,'training_fold_metrics':training_metrics,
                                   'validation_fold_metrics':metrics,'median_validation_sharpe':float(median(finite))})
        if not validation: raise ValueError(f'No eligible nested cointegration candidate in outer run {outer_run}')
        validation.sort(key=lambda row:(row['median_validation_sharpe'],np.mean([m['total_return'] for m in row['validation_fold_metrics']])),reverse=True)
        selected=validation[0]
        held=prices.index[slice(*held_range)]
        metric=evaluate(tuple(selected['pair']),selected['parameters'],held)
        runs.append({'run':outer_run,'inner_folds':folds,'outer_held_out':list(held_range),'pairs_considered':len(pairs),
                     'parameter_pair_candidates':len(pairs)*len(PARAMETERS),'training_eligible':len(training_rank),
                     'validation_shortlist':min(SHORTLIST,len(training_rank)),'validation_eligible':len(validation),
                     'selected':selected,'held_out':metric,
                     'held_out_passed':bool(metric['total_return']>0 and metric['sharpe']>0)})
        print(f'outer run {outer_run}/5 complete',flush=True)
    metric_names=('total_return','sharpe','alpha','max_drawdown')
    output={'test':'Pure singleton-pair cointegration with nested chronological validation','strategies':['cointegration'],
            'regime_filter':None,'portfolio_weighting':'rolling hedge-ratio pair normalized to unit gross',
            'pair_screening':{'frozen_initial_training_fdr_screen':True,'fdr_alpha':.05,'survivors':len(pairs)},
            'parameter_optimizer':'exhaustive grid followed by median inner-validation Sharpe',
            'parameter_grid':PARAMETERS,'execution':{'execution_delay_bars':1,'fee_per_order':FEE,'slippage_per_order':SLIPPAGE},
            'runs':runs,'averages':{name:float(np.mean([r['held_out'][name] for r in runs])) for name in metric_names},
            'passed_runs':sum(r['held_out_passed'] for r in runs),'total_runs':5,
            'scientific_status':'Diagnostic: these historical held-out intervals were viewed earlier.'}
    path=ROOT/'outputs'/'checkpoint_pure_cointegration_nested_five_outer_summary.json'
    path.write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(output,indent=2,allow_nan=False))

if __name__=='__main__': main()
