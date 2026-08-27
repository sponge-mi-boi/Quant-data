"""Equal-weight CMV/MT/CMT using validation-selected parameters, without regimes."""
import json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd
os.environ.setdefault('NUMBA_DISABLE_JIT','1')
ROOT = Path(r"path\to\root"); PROJECT=ROOT/'work'/'PythonProject1_basicbacktester'/'Published'; sys.path[:0]=[str(PROJECT/'src'),str(ROOT/'work')]
from src.data_filter import get_time_period
from src.strategies import _get_signals_mv_cross_asset,_get_signals_momentum_tr,_get_signals_momentum_cross_asset
from run_cmv_full_three_stage_five_cycles import CYCLES,FEE,SLIPPAGE,performance
from run_cross_momentum_timeseries_momentum_rule_regime import normalize,net_returns,passed

ONLY_STRATEGY=os.environ.get('ONLY_STRATEGY')
STRATEGY_SET=os.environ.get('STRATEGY_SET')
ROLLING_FIXED=os.environ.get('ROLLING_FIXED_WINDOWS','false').lower()=='true'
SLEEVES=(('cross_asset_mv','momentum_trending') if STRATEGY_SET=='cmv_mt' else
         ((ONLY_STRATEGY,) if ONLY_STRATEGY else
          ('cross_asset_mv','momentum_trending','cross_asset_momentum_trending')))
def main():
    source=json.loads((ROOT/'outputs'/('checkpoint_rolling_fixed_500_260_260_three_strategy_corr_liq_disp_rbf_svm_summary.json' if ROLLING_FIXED else 'checkpoint_three_strategy_parameter_optimized_ac_vol_corr_hmm_summary.json')).read_text())
    universe=pd.read_parquet(PROJECT/'data'/'processed'/'close_1d_10y.parquet').columns.tolist(); prices=get_time_period(universe,time_peri=(0,2060)); returns=prices.pct_change().fillna(0.); market=get_time_period(['SPY'],time_peri=(0,2060)).reindex(prices.index)['SPY'].pct_change().fillna(0.)
    cache={}; runs=[]
    def signal(name,p):
        key=(name,tuple(sorted(p.items())))
        if key not in cache:
            config={'stock_list':universe,'time_period':(0,2060),'freq':'d','strat_class':{name:p},'parameters_':p}
            cache[key]=(_get_signals_mv_cross_asset(config) if name=='cross_asset_mv' else (_get_signals_momentum_tr(config,prices) if name=='momentum_trending' else _get_signals_momentum_cross_asset(config))).reindex(prices.index).fillna(0.)
        return cache[key]
    cycles=([{'training':(260*i,500+260*i),'validation':(500+260*i,760+260*i),'held_out':(760+260*i,1020+260*i)} for i in range(5)] if ROLLING_FIXED else CYCLES)
    for number,(periods,prior) in enumerate(zip(cycles,source['runs']),1):
        selected=prior['selected_strategies']; sleeves={name:normalize(signal(name,selected[name]['parameters']),selected[name]['assets']) for name in SLEEVES}
        columns=list(dict.fromkeys(c for frame in sleeves.values() for c in frame.columns)); target=pd.DataFrame(0.,index=prices.index,columns=columns)
        for name,frame in sleeves.items(): target.loc[:,frame.columns]+=frame/len(SLEEVES)
        held=prices.index[slice(*periods['held_out'])]; metric=performance(net_returns(target,returns).reindex(held).fillna(0.),market.reindex(held)); active=target.reindex(held).abs().sum(axis=1).clip(upper=1.)
        runs.append({'run':number,'selected_strategies':{name:selected[name] for name in SLEEVES},'allocation':{name:1/len(SLEEVES) for name in SLEEVES},'average_implicit_cash':float((1-active).mean()),'held_out':metric,'held_out_passed':passed(metric)})
    labels={'cross_asset_mv':'CMV only','momentum_trending':'time-series momentum only',
            'cross_asset_momentum_trending':'cross-sectional momentum only'}
    label=('CMV + MT, equal weight' if STRATEGY_SET=='cmv_mt' else
           labels.get(ONLY_STRATEGY,'CMV + MT + CMT, equal weight'))
    names=('total_return','sharpe','alpha','max_drawdown'); output={'test':f'Parameter-optimized {label}, no regime','classifier':None,'regime_filter':None,'allocation':{name:1/len(SLEEVES) for name in SLEEVES},'execution':{'execution_delay_bars':1,'fee_per_order':FEE,'slippage_per_order':SLIPPAGE},'runs':runs,'average_held_out_metrics':{n:float(np.mean([r['held_out'][n] for r in runs])) for n in names},'held_out_pass_count':sum(r['held_out_passed'] for r in runs),'scientific_status':'Diagnostic: these historical windows were viewed earlier.'}
    output_names={'cross_asset_mv':'checkpoint_cmv_parameter_optimized_no_regime_summary.json',
                  'momentum_trending':'checkpoint_mt_parameter_optimized_no_regime_summary.json',
                  'cross_asset_momentum_trending':'checkpoint_cmt_parameter_optimized_no_regime_summary.json'}
    output_name=('checkpoint_cmv_mt_parameter_optimized_no_regime_summary.json'
                 if STRATEGY_SET=='cmv_mt' else
                 output_names.get(ONLY_STRATEGY,'checkpoint_three_strategy_parameter_optimized_no_regime_summary.json'))
    if ROLLING_FIXED:
        rolling_names={'cross_asset_mv':'cmv','momentum_trending':'mt','cross_asset_momentum_trending':'cmt'}
        output_name=(f'checkpoint_rolling_fixed_500_260_260_{rolling_names[ONLY_STRATEGY]}_parameter_optimized_no_regime_summary.json'
                     if ONLY_STRATEGY else
                     ('checkpoint_rolling_fixed_500_260_260_cmv_mt_parameter_optimized_no_regime_summary.json'
                      if STRATEGY_SET=='cmv_mt' else
                      'checkpoint_rolling_fixed_500_260_260_three_strategy_equal_weight_no_regime_summary.json'))
    path=ROOT/'outputs'/output_name; path.write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8'); print(json.dumps(output,indent=2,allow_nan=False))
if __name__=='__main__': main()
