"""Pure singleton cointegration with nested Adam/SPSA parameter optimization."""
import json, os, sys
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

os.environ.setdefault('NUMBA_DISABLE_JIT','1')
ROOT = Path (__file__).resolve().parent.parent 
PROJECT=ROOT 
sys.path[:0]=[str(PROJECT/'src/quant_backtester'),str(ROOT/'artifacts')]
from src.quant_backtester.strategies import _get_signals
from run_cmv_full_three_stage_five_cycles import FEE,SLIPPAGE,performance
from run_cointegration_all_pairs import benjamini_hochberg

FOLDS=[{'training':(0,280),'validation':(280,400)},
       {'training':(120,400),'validation':(400,520)},
       {'training':(240,520),'validation':(520,640)}]
OUTER_HELD=(760,1020)
COARSE=[{'roll':roll,'z_threshold':(entry,0.)}
        for roll in (20,32,60) for entry in (1.5,2.,2.5)]
PAIR_SHORTLIST=10
ITERATIONS=60
RESTARTS=2

def decode(theta):
    s=1/(1+np.exp(-np.clip(theta,-20,20)))
    roll=int(round(20+40*s[0]))
    entry=float(1.25+1.75*s[1])
    exit_=float((entry-.10)*.75*s[2])
    return {'roll':roll,'z_threshold':(entry,exit_)}

def adam_spsa(objective,seed):
    rng=np.random.default_rng(seed); theta=rng.normal(0,.35,3); m=np.zeros(3); v=np.zeros(3)
    best=(objective(theta),theta.copy())
    for step_number in range(1,ITERATIONS+1):
        delta=rng.choice((-1.,1.),3); c=.12/(step_number**.101)
        grad=(objective(theta+c*delta)-objective(theta-c*delta))/(2*c)*delta
        m=.9*m+.1*grad; v=.999*v+.001*grad*grad
        step=.06*np.sqrt(1-.999**step_number)/(1-.9**step_number)
        theta+=step*m/(np.sqrt(v)+1e-8)
        value=objective(theta)
        if value>best[0]: best=(value,theta.copy())
    return best

def main():
    all_prices=pd.read_parquet(PROJECT/'data'/'close_1d_10y.parquet').iloc[:2060]
    market=all_prices['SPY'].pct_change().fillna(0.); prices=all_prices.drop(columns='SPY')
    returns=prices.pct_change().fillna(0.)
    screen=pd.read_csv(ROOT/'artifacts'/'checkpoint_costs_cointegration_test1_all_pair_pvalues.csv')
    screen['q_value']=benjamini_hochberg(screen.p_value.to_numpy())
    pairs=[tuple(row) for row in screen.loc[screen.q_value<=.05,['asset_1','asset_2']].itertuples(index=False,name=None)]
    cache={}
    def series(pair,params):
        entry,exit_=params['z_threshold']; key=(pair,params['roll'],round(entry,8),round(exit_,8))
        if key not in cache:
            raw=_get_signals({'stock_list':list(pair),'parameters_':params,'weights_filter':{}},prices[list(pair)]).reindex(prices.index).fillna(0.)
            target=raw.div(raw.abs().sum(axis=1).replace(0,np.nan),axis=0).fillna(0.)
            executed=target.shift(1).fillna(0.); turnover=executed.diff().abs().sum(axis=1).fillna(executed.abs().sum(axis=1))
            net=(executed*returns[list(pair)]).sum(axis=1)-turnover*(FEE+SLIPPAGE)
            changes=executed.diff().abs().sum(axis=1).gt(1e-12); cache[key]=(net,changes)
        return cache[key]
    def evaluate(pair,params,index):
        net,changes=series(pair,params); result=performance(net.reindex(index).fillna(0.),market.reindex(index))
        result['position_changes']=int(changes.reindex(index).fillna(False).sum()); return result
    runs=[]
    for outer_run in range(1,6):
        offset=260*(outer_run-1)
        folds=[{k:(a+offset,b+offset) for k,(a,b) in fold.items()} for fold in FOLDS]
        held_range=(OUTER_HELD[0]+offset,OUTER_HELD[1]+offset)
        ranked=[]
        for pair in pairs:
            best=None
            for params in COARSE:
                metrics=[evaluate(pair,params,prices.index[slice(*fold['training'])]) for fold in folds]
                if all(np.isfinite(x['sharpe']) and x['position_changes']>=3 for x in metrics):
                    score=float(median(x['sharpe'] for x in metrics))
                    if best is None or score>best[0]: best=(score,params,metrics)
            if best is not None: ranked.append((best[0],pair,best[1],best[2]))
        ranked.sort(key=lambda row:row[0],reverse=True); shortlist=ranked[:PAIR_SHORTLIST]
        optimized=[]
        for pair_number,(_,pair,coarse_params,training_metrics) in enumerate(shortlist):
            validation_indices=[prices.index[slice(*fold['validation'])] for fold in folds]
            def objective(theta):
                params=decode(theta); metrics=[evaluate(pair,params,index) for index in validation_indices]
                if not all(np.isfinite(x['sharpe']) and x['position_changes']>=2 for x in metrics): return -1e6
                return float(median(x['sharpe'] for x in metrics))
            for restart in range(RESTARTS):
                score,theta=adam_spsa(objective,50000+1000*outer_run+100*pair_number+restart)
                params=decode(theta); metrics=[evaluate(pair,params,index) for index in validation_indices]
                optimized.append({'pair':list(pair),'parameters':params,'training_coarse_parameters':coarse_params,
                                  'training_fold_metrics':training_metrics,'validation_fold_metrics':metrics,
                                  'median_validation_sharpe':score,'restart':restart})
        optimized.sort(key=lambda row:(row['median_validation_sharpe'],np.mean([m['total_return'] for m in row['validation_fold_metrics']])),reverse=True)
        selected=optimized[0]; held=prices.index[slice(*held_range)]
        metric=evaluate(tuple(selected['pair']),selected['parameters'],held)
        runs.append({'run':outer_run,'inner_folds':folds,'outer_held_out':list(held_range),'pair_shortlist_size':len(shortlist),
                     'adam_candidates':len(optimized),'selected':selected,'held_out':metric,
                     'held_out_passed':bool(metric['total_return']>0 and metric['sharpe']>0)})
        print(f'outer run {outer_run}/5 complete',flush=True)
    names=('total_return','sharpe','alpha','max_drawdown')
    output={'test':'Pure singleton-pair cointegration with nested Adam/SPSA optimization','strategies':['cointegration'],
            'regime_filter':None,'pair_screening':{'frozen_initial_training_fdr_screen':True,'fdr_alpha':.05,'survivors':len(pairs)},
            'pair_selection':'top ten by median inner-training Sharpe using coarse grid; Adam applied separately to each pair',
            'optimizer':'Adam with deterministic SPSA gradients','adam_iterations':ITERATIONS,'restarts_per_pair':RESTARTS,
            'parameter_bounds':{'roll':[20,60],'entry':[1.25,3.0],'exit':'0 to 75% of entry minus 0.10'},
            'execution':{'execution_delay_bars':1,'fee_per_order':FEE,'slippage_per_order':SLIPPAGE},
            'runs':runs,'averages':{name:float(np.mean([r['held_out'][name] for r in runs])) for name in names},
            'passed_runs':sum(r['held_out_passed'] for r in runs),'total_runs':5,
            'scientific_status':'Diagnostic: these historical held-out intervals were viewed earlier.'}
    path=ROOT/'artifacts'/'checkpoint_pure_cointegration_adam_nested_five_outer_summary.json'
    path.write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8'); print(json.dumps(output,indent=2,allow_nan=False))

if __name__=='__main__': main()
