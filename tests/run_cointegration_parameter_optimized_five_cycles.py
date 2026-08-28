"""Five-cycle pure cointegration with validation-selected pair and parameters."""
import json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault('NUMBA_DISABLE_JIT','1')
ROOT = Path (__file__).resolve().parent.parent 
PROJECT=ROOT; sys.path[:0]=[str(PROJECT/'src/quant_backtester'),str(ROOT/'artifacts')]
from quant_backtester .strategies import _get_signals
from run_cmv_full_three_stage_five_cycles import CYCLES,FEE,SLIPPAGE,performance
from run_cointegration_all_pairs import benjamini_hochberg

ENTRY_THRESHOLDS=(1.5,2.,2.5)
EXIT_THRESHOLDS=(0.,.5,1.)
PARAMETERS=[{'z_threshold':(entry,exit_),'roll':roll}
            for roll in (20,32,60) for entry in ENTRY_THRESHOLDS
            for exit_ in EXIT_THRESHOLDS if exit_<entry]
COMBINE_TOP=os.environ.get('COMBINE_TOP_PAIRS','false').lower()=='true'
TOP_PAIR_COUNT=int(os.environ.get('TOP_PAIR_COUNT','0'))
BETA_NEUTRAL=os.environ.get('COINTEGRATION_BETA_NEUTRAL','false').lower()=='true'
BETA_ROLL=60

def main():
    all_prices=pd.read_parquet(PROJECT/'data'/'close_1d_10y.parquet').iloc[:2060]
    market=all_prices['SPY'].pct_change().fillna(0.); prices=all_prices.drop(columns='SPY'); returns=prices.pct_change().fillna(0.)
    screen=pd.read_csv(ROOT/'artifacts'/'checkpoint_costs_cointegration_test1_all_pair_pvalues.csv'); screen['q_value']=benjamini_hochberg(screen.p_value.to_numpy())
    pairs=[tuple(x) for x in screen.loc[screen.q_value<=.05,['asset_1','asset_2']].itertuples(index=False,name=None)]
    cache={}
    def series(pair,p):
        key=(pair,p['z_threshold'],p['roll'])
        if key not in cache:
            cfg={'stock_list':list(pair),'parameters_':p,'weights_filter':{}}
            raw=_get_signals(cfg,prices[list(pair)]).reindex(prices.index).fillna(0.)
            target=raw.div(raw.abs().sum(axis=1).replace(0,np.nan),axis=0).fillna(0.); executed=target.shift(1).fillna(0.)
            turnover=executed.diff().abs().sum(axis=1).fillna(executed.abs().sum(axis=1))
            net=(executed*returns[list(pair)]).sum(axis=1)-turnover*(FEE+SLIPPAGE)
            changes=executed.diff().abs().sum(axis=1).gt(1e-12)
            cache[key]=(net,changes,target)
        return cache[key]
    def evaluate(pair,p,index):
        net,changes,_=series(pair,p); metric=performance(net.reindex(index).fillna(0.),market.reindex(index)); metric['position_changes']=int(changes.reindex(index).fillna(False).sum()); return metric
    def evaluate_portfolio(members,index):
        if not BETA_NEUTRAL:
            net=pd.concat([series(tuple(x['pair']),x['parameters'])[0] for x in members],axis=1).mean(axis=1)
            return performance(net.reindex(index).fillna(0.),market.reindex(index))
        targets=[series(tuple(x['pair']),x['parameters'])[2] for x in members]
        columns=list(dict.fromkeys(c for frame in targets for c in frame.columns)); target=pd.DataFrame(0.,index=prices.index,columns=columns)
        for frame in targets: target.loc[:,frame.columns]+=frame/len(targets)
        r=returns[columns]; market_var=market.rolling(BETA_ROLL,min_periods=BETA_ROLL).var(); beta=r.rolling(BETA_ROLL,min_periods=BETA_ROLL).cov(market).div(market_var,axis=0)
        values=target.to_numpy(); b=beta.to_numpy(); projected=np.zeros_like(values)
        for t in range(BETA_ROLL-1,len(target)):
            if not np.isfinite(b[t]).all(): continue
            projected[t]=values[t]-b[t]*(b[t]@values[t])/max(float(b[t]@b[t]),1e-12)
            gross=np.abs(projected[t]).sum(); desired=np.abs(values[t]).sum()
            if gross>1e-12: projected[t]*=desired/gross
        neutral=pd.DataFrame(projected,index=target.index,columns=columns); executed=neutral.shift(1).fillna(0.); turnover=executed.diff().abs().sum(axis=1).fillna(executed.abs().sum(axis=1)); net=(executed*r).sum(axis=1)-turnover*(FEE+SLIPPAGE)
        metric=performance(net.reindex(index).fillna(0.),market.reindex(index)); residual=np.nansum(neutral.reindex(index).to_numpy()*beta.reindex(index).to_numpy(),axis=1); metric['max_abs_beta_residual']=float(np.nanmax(np.abs(residual))); return metric
    runs=[]
    for number,periods in enumerate(CYCLES,1):
        train=prices.index[slice(*periods['training'])]; val=prices.index[slice(*periods['validation'])]; held=prices.index[slice(*periods['held_out'])]
        shortlist=[]; training_tested=0
        for p in PARAMETERS:
            ranked=[]
            for pair in pairs:
                metric=evaluate(pair,p,train); training_tested+=1
                if np.isfinite(metric['sharpe']) and metric['position_changes']>=5: ranked.append((pair,metric))
            ranked.sort(key=lambda x:(x[1]['sharpe'],x[1]['total_return']),reverse=True)
            shortlist.extend((pair,p,m) for pair,m in ranked[:10])
        validation=[]
        for pair,p,train_metric in shortlist:
            metric=evaluate(pair,p,val)
            if np.isfinite(metric['sharpe']) and metric['position_changes']>=3:
                validation.append({'pair':list(pair),'parameters':p,'training':train_metric,'validation':metric})
        if not validation: raise ValueError(f'no validation eligible candidate in run {number}')
        validation.sort(key=lambda x:(x['validation']['sharpe'],x['validation']['total_return']),reverse=True)
        if COMBINE_TOP:
            disjoint=[]; used=set(); seen_pairs=set()
            for item in validation:
                pair_key=tuple(sorted(item['pair']))
                if TOP_PAIR_COUNT:
                    if pair_key not in seen_pairs:
                        disjoint.append(item); seen_pairs.add(pair_key)
                elif not used.intersection(item['pair']):
                    disjoint.append(item); used.update(item['pair'])
                if len(disjoint)>=max(10,TOP_PAIR_COUNT): break
            portfolios=[]
            sizes=([TOP_PAIR_COUNT] if TOP_PAIR_COUNT else range(2,min(5,len(disjoint))+1))
            for size in sizes:
                if size>len(disjoint): continue
                members=disjoint[:size]; portfolios.append({'size':size,'members':members,'validation':evaluate_portfolio(members,val)})
            if not portfolios: raise ValueError(f'fewer than {TOP_PAIR_COUNT or 2} eligible validation pairs in run {number}')
            portfolios.sort(key=lambda x:(x['validation']['sharpe'],x['validation']['total_return']),reverse=True); selected=portfolios[0]
            held_metric=evaluate_portfolio(selected['members'],held)
        else:
            selected=validation[0]; held_metric=evaluate(tuple(selected['pair']),selected['parameters'],held)
        runs.append({'run':number,'pairs_screened':len(pairs),'training_parameter_pair_candidates':training_tested,'validation_candidates':len(validation),'selected':selected,'held_out':held_metric,'held_out_passed':bool(held_metric['total_return']>0 and held_metric['sharpe']>0)})
        print(f'cycle {number}/5 complete',flush=True)
    names=('total_return','sharpe','alpha','max_drawdown')
    combined_label=(f'Pure cointegration: top {TOP_PAIR_COUNT} validation-selected unique pairs'
                    if TOP_PAIR_COUNT else
                    ('Beta-neutral combined top disjoint cointegration pairs'
                     if BETA_NEUTRAL else 'Combined top disjoint cointegration pairs'))
    output={'test':combined_label if COMBINE_TOP else 'Pure cointegration: five-cycle parameter and pair optimization','strategies':['cointegration'],'portfolio_weighting':'equal weight across pairs' if COMBINE_TOP else 'singleton pair','neutrality':['rolling_market_beta'] if BETA_NEUTRAL else [],'beta_lookback_bars':BETA_ROLL if BETA_NEUTRAL else None,'regime_filter':None,'parameter_optimizer':'exhaustive validation grid','parameter_grid':PARAMETERS,'pair_screening':{'screened_all_pairs':105111,'fdr_alpha':.05,'fdr_survivors':len(pairs),'screen_data':'initial training only'},'execution':{'execution_delay_bars':1,'fee_per_order':FEE,'slippage_per_order':SLIPPAGE},'runs':runs,'average_held_out_metrics':{n:float(np.mean([r['held_out'][n] for r in runs])) for n in names},'held_out_pass_count':sum(r['held_out_passed'] for r in runs),'scientific_status':'Diagnostic: these historical held-out windows were viewed earlier.'}
    filename=(f'checkpoint_pure_cointegration_top_{TOP_PAIR_COUNT}_pairs_entry_exit_roll_optimized_five_cycles_summary.json'
              if COMBINE_TOP and TOP_PAIR_COUNT else
              (('checkpoint_combined_top_cointegration_pairs_beta_neutral_parameter_optimized_five_cycles_summary.json'
                if BETA_NEUTRAL else 'checkpoint_combined_top_cointegration_pairs_parameter_optimized_five_cycles_summary.json')
               if COMBINE_TOP else 'checkpoint_pure_cointegration_parameter_optimized_five_cycles_summary.json'))
    path=ROOT/'artifacts'/filename; path.write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8'); print(json.dumps(output,indent=2,allow_nan=False))

if __name__=='__main__': main()
