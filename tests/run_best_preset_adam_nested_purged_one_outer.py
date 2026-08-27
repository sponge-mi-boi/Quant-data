"""One outer test of the best AC/correlation/volatility preset allocator with nested validation."""
import json, os, pickle, sys
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd

os.environ.setdefault('NUMBA_DISABLE_JIT','1')
PRESET_FEATURE_SET=os.environ.get('NESTED_PRESET_FEATURE_SET','ac_corr_vol').lower()
PRESET_STATE_MODE=os.environ.get('NESTED_PRESET_STATE_MODE','continuous').lower()
FEATURE_COMBO_RAW=os.environ.get('NESTED_PRESET_FEATURE_COMBO','').strip().lower()
FEATURE_COMBO=tuple(x.strip() for x in FEATURE_COMBO_RAW.split(',') if x.strip())
VALID_COMBO_FEATURES={'corr','liq','ac','dis','var'}
if FEATURE_COMBO:
    if len(FEATURE_COMBO) not in {4,5} or len(set(FEATURE_COMBO))!=len(FEATURE_COMBO):
        raise ValueError('NESTED_PRESET_FEATURE_COMBO must contain four or five unique feature names')
    unknown=set(FEATURE_COMBO)-VALID_COMBO_FEATURES
    if unknown:
        raise ValueError(f'Unknown preset feature names: {sorted(unknown)}')
    PRESET_STATE_MODE='combinatorial'
if PRESET_FEATURE_SET not in {'ac_corr_vol','disp_corr_liq'}:
    raise ValueError(
        'NESTED_PRESET_FEATURE_SET must be ac_corr_vol or disp_corr_liq; '
        f'got {PRESET_FEATURE_SET!r}'
    )
os.environ['PRESET_FEATURE_SET']=PRESET_FEATURE_SET
if PRESET_STATE_MODE not in {'continuous','discrete8','combinatorial'}:
    raise ValueError(
        'NESTED_PRESET_STATE_MODE must be continuous or discrete8; '
        f'got {PRESET_STATE_MODE!r}'
    )
if PRESET_STATE_MODE=='discrete8' and PRESET_FEATURE_SET!='disp_corr_liq':
    raise ValueError('discrete8 currently requires NESTED_PRESET_FEATURE_SET=disp_corr_liq')
ROOT = Path(r"path\to\root"); PROJECT=ROOT/'work'/'PythonProject1_basicbacktester'/'Published'
sys.path[:0]=[str(PROJECT/'src'),str(ROOT/'work')]
from src.data_filter import get_time_period
from src.hmm_regime import build_hmm_features
from run_cmv_mt_cmt_rule_regime import build_correlation_liquidity_dispersion_features
from src.strategies import (_get_signals_mv_cross_asset, _get_signals_momentum_tr, _get_signals_momentum_cross_asset)
from run_cmv_full_three_stage_five_cycles import FEE,SLIPPAGE,performance
from run_cross_momentum_timeseries_momentum_rule_regime import normalize,net_returns,passed
from run_cmv_mt_cmt_bayesian_hmm_regime import select_strategy_on_validation
from run_three_strategy_adam_disp_corr_liq import allocations,adam_spsa,decode,combine,SLEEVES

FOLDS=[
 {'strategy_training':(0,280),'strategy_validation':(280,400),'filter_validation':(400,520)},
 {'strategy_training':(120,400),'strategy_validation':(400,520),'filter_validation':(520,640)},
 {'strategy_training':(240,520),'strategy_validation':(520,640),'filter_validation':(640,760)},
]
OUTER_HELD=(760,1020); REBALANCES=(5,10,20)
OUTER_RUN=int(os.environ.get('NESTED_OUTER_RUN','1'))

def combinatorial_allocations(binary,half,rebalance,cmv_scale,mt_scale,cmt_scale,cash_scale):
    """Fixed, transparent economic scores for any supported 4/5-feature state."""
    f=binary.dropna(); score=pd.DataFrame(.05,index=f.index,columns=[*SLEEVES,'cash'])
    if 'corr' in f:
        hi=f['corr']; lo=1-hi; score['momentum_trending']+=hi; score['cross_asset_mv']+=lo; score['cross_asset_momentum_trending']+=lo
    if 'liq' in f:
        hi=f['liq']; lo=1-hi; score['momentum_trending']+=.6*hi; score['cross_asset_momentum_trending']+=.8*hi; score['cash']+=1.5*lo; score['cross_asset_mv']+=.3*lo
    if 'ac' in f:
        hi=f['ac']; lo=1-hi; score['momentum_trending']+=2*hi; score['cross_asset_momentum_trending']+=2*hi; score['cross_asset_mv']+=2*lo
    if 'dis' in f:
        hi=f['dis']; lo=1-hi; score['cross_asset_mv']+=hi; score['cross_asset_momentum_trending']+=hi; score['momentum_trending']+=lo
    if 'var' in f:
        hi=f['var']; lo=1-hi; score['cash']+=1.2*hi; score['cross_asset_mv']+=.4*hi; score['momentum_trending']+=.3*lo; score['cross_asset_momentum_trending']+=.3*lo
    if 'corr' in f and 'dis' in f:
        score['cash']+=.3*f['corr']*f['dis']
    score=score.mul(pd.Series({'cross_asset_mv':cmv_scale,'momentum_trending':mt_scale,'cross_asset_momentum_trending':cmt_scale,'cash':cash_scale}))
    weight=score.div(score.sum(axis=1),axis=0).ewm(halflife=half,adjust=False).mean()
    weight.loc[np.arange(len(weight))%rebalance!=0]=np.nan
    return weight.ffill().fillna(0.)

def main():
    offset=260*(OUTER_RUN-1)
    folds=[{k:(a+offset,b+offset) for k,(a,b) in fold.items()} for fold in FOLDS]
    outer_held=(OUTER_HELD[0]+offset,OUTER_HELD[1]+offset)
    universe=pd.read_parquet(PROJECT/'data'/'processed'/'close_1d_10y.parquet').columns.tolist()
    prices=get_time_period(universe,time_peri=(0,2060)); returns=prices.pct_change().fillna(0.)
    market=get_time_period(['SPY'],time_peri=(0,2060)).reindex(prices.index)['SPY'].pct_change().fillna(0.)
    if FEATURE_COMBO:
        volume=pd.read_parquet(PROJECT/'data'/'processed'/'volume_1d_10y.parquet',columns=universe).reindex(prices.index)
        cld=build_correlation_liquidity_dispersion_features(prices,volume,returns)
        hmm=build_hmm_features(market,returns)
        features=pd.concat([cld[['corr','liq','dis']],hmm[['ac','var']]],axis=1).dropna()
        feature_names=[{'corr':'correlation','liq':'liquidity','ac':'autocorrelation','dis':'dispersion','var':'volatility'}[x] for x in FEATURE_COMBO]
        output_slug='combo_'+'_'.join(FEATURE_COMBO)
    elif PRESET_FEATURE_SET=='disp_corr_liq':
        volume=pd.read_parquet(PROJECT/'data'/'processed'/'volume_1d_10y.parquet',columns=universe).reindex(prices.index)
        features=build_correlation_liquidity_dispersion_features(prices,volume,returns)
        feature_names=['correlation','liquidity','dispersion']
        output_slug='corr_liq_disp'
    else:
        features=build_hmm_features(market,returns).loc[:,['ac','var','corr']]
        feature_names=['autocorrelation','volatility','correlation']
        output_slug='ac_corr_vol'
    allocation_features=features
    state_definitions=None
    if PRESET_STATE_MODE=='combinatorial':
        allocation_features=features.loc[:,FEATURE_COMBO].copy()
        for column in FEATURE_COMBO:
            threshold=0. if column=='ac' else .5
            allocation_features[column]=(allocation_features[column]>=threshold).astype(float)
        state_definitions=[dict(zip(FEATURE_COMBO,map(bool,state))) for state in product((0,1),repeat=len(FEATURE_COMBO))]
    elif PRESET_STATE_MODE=='discrete8':
        allocation_features=features.copy()
        for column in ('corr','liq','dis'):
            allocation_features[column]=(allocation_features[column]>=.5).astype(float)
        state_definitions=[
            {'state':f'corr_{"high" if corr else "low"}__liq_{"high" if liq else "low"}__disp_{"high" if dis else "low"}',
             'corr_high':bool(corr),'liq_high':bool(liq),'disp_high':bool(dis)}
            for corr,liq,dis in product((0,1),repeat=3)
        ]
        output_slug+='__discrete8'
    def cfg(name,p): return {'stock_list':universe,'time_period':(0,2060),'freq':'d','strat_class':{name:p},'parameters_':p}
    grids={'cross_asset_mv':[{'z_threshold':z} for z in (1.5,2.,2.5)],
           'momentum_trending':[{'z_threshold':z,'roll':r} for r,z in product((20,30,60),(1.5,2.,2.5))],
           'cross_asset_momentum_trending':[{'z_threshold':z,'roll':r} for r,z in product((20,35,60),(1.5,2.,2.5))]}
    cache_dir=ROOT/'work'/'.cache'; cache_dir.mkdir(exist_ok=True)
    cache_path=cache_dir/f'nested_three_sleeves_v1_outer_{OUTER_RUN}.pkl'
    if cache_path.exists():
        with cache_path.open('rb') as handle: prepared=pickle.load(handle)
        print(f'loaded shared strategy cache for outer run {OUTER_RUN}',flush=True)
    else:
        candidate_raw={n:[] for n in SLEEVES}
        for name,grid in grids.items():
            for p in grid:
                c=cfg(name,p); raw=(_get_signals_mv_cross_asset(c) if name=='cross_asset_mv' else (_get_signals_momentum_tr(c,prices) if name=='momentum_trending' else _get_signals_momentum_cross_asset(c)))
                candidate_raw[name].append((p,raw.reindex(prices.index).fillna(0.)))
        prepared=[]
        for number,f in enumerate(folds,1):
            periods={'training':f['strategy_training'],'validation':f['strategy_validation']}; selected={}; sleeves={}
            for name in SLEEVES:
                selected[name],_=select_strategy_on_validation(name,candidate_raw[name],returns,market,prices.index,periods)
                raw=next(frame for p,frame in candidate_raw[name] if p==selected[name]['parameters'])
                sleeves[name]=normalize(raw,selected[name]['assets'])
            prepared.append({'fold':number,'spec':f,'selected_strategies':selected,'sleeves':sleeves})
            print(f'inner strategy fold {number}/3 complete',flush=True)
        with cache_path.open('wb') as handle: pickle.dump(prepared,handle,pickle.HIGHEST_PROTOCOL)
        print(f'saved shared strategy cache for outer run {OUTER_RUN}',flush=True)
    choices=[]
    for rebalance in REBALANCES:
        def objective(theta):
            half,cmv_s,mt_s,cmt_s,cash_s=decode(theta); values=[]
            for item in prepared:
                a=(combinatorial_allocations(allocation_features,half,rebalance,cmv_s,mt_s,cmt_s,cash_s) if FEATURE_COMBO else allocations(allocation_features,half,rebalance,cmv_s,mt_s,cmt_s,cash_s))
                idx=prices.index[slice(*item['spec']['filter_validation'])]
                m=performance(net_returns(combine(item['sleeves'],a,prices.index),returns).reindex(idx).fillna(0.),market.reindex(idx))
                values.append(m['sharpe'] if np.isfinite(m['sharpe']) else -1e6)
            return float(np.median(values))
        for restart in range(2):
            _,theta=adam_spsa(objective,19000+100*rebalance+restart); half,cmv_s,mt_s,cmt_s,cash_s=decode(theta); fold_metrics=[]
            for item in prepared:
                a=(combinatorial_allocations(allocation_features,half,rebalance,cmv_s,mt_s,cmt_s,cash_s) if FEATURE_COMBO else allocations(allocation_features,half,rebalance,cmv_s,mt_s,cmt_s,cash_s)); idx=prices.index[slice(*item['spec']['filter_validation'])]
                fold_metrics.append(performance(net_returns(combine(item['sleeves'],a,prices.index),returns).reindex(idx).fillna(0.),market.reindex(idx)))
            choices.append({'smoothing_half_life':half,'rebalance_every_bars':rebalance,'cmv_scale':cmv_s,'mt_scale':mt_s,'cmt_scale':cmt_s,'cash_scale':cash_s,'inner_fold_metrics':fold_metrics,'median_inner_sharpe':float(np.median([m['sharpe'] for m in fold_metrics]))})
        print(f'Adam rebalance candidate {rebalance} complete',flush=True)
    choices.sort(key=lambda x:x['median_inner_sharpe'],reverse=True); selected=choices[0]; final=prepared[-1]
    alloc=(combinatorial_allocations(allocation_features,selected['smoothing_half_life'],selected['rebalance_every_bars'],selected['cmv_scale'],selected['mt_scale'],selected['cmt_scale'],selected['cash_scale']) if FEATURE_COMBO else allocations(allocation_features,selected['smoothing_half_life'],selected['rebalance_every_bars'],selected['cmv_scale'],selected['mt_scale'],selected['cmt_scale'],selected['cash_scale']))
    held=prices.index[slice(*outer_held)]; metric=performance(net_returns(combine(final['sleeves'],alloc,prices.index),returns).reindex(held).fillna(0.),market.reindex(held))
    output={'test':f'No-learning combinatorial preset {"/".join(feature_names)} + Adam, nested outer run {OUTER_RUN}','outer_run':OUTER_RUN,'features':feature_names,'feature_codes':list(FEATURE_COMBO) if FEATURE_COMBO else None,'state_mode':PRESET_STATE_MODE,'state_thresholds':({'autocorrelation':0.,**{name:.5 for name in feature_names if name!='autocorrelation'}} if FEATURE_COMBO else (.5 if PRESET_STATE_MODE=='discrete8' else None)),'state_definitions':state_definitions,'inner_folds':folds,'outer_held_out':list(outer_held),'purge_bars':0,'purge_note':'Preset rules have no forward-return labels, so target-overlap purging is not applicable; strategy and filter validation blocks are non-overlapping.','strategy_selection':'separate earlier validation block per fold','filter_selection':'median filter-validation Sharpe across three chronological folds','optimizer':'Adam with SPSA validation gradients','classifier':None,'learned_regime_mapping':False,'selected_filter':selected,'final_selected_strategies':final['selected_strategies'],'held_out':metric,'held_out_passed':passed(metric),'execution':{'execution_delay_bars':1,'fee_per_order':FEE,'slippage_per_order':SLIPPAGE},'scientific_status':'Diagnostic: this historical held-out interval was viewed earlier.'}
    path=ROOT/'outputs'/f'checkpoint_preset_{output_slug}_adam_nested_outer_run_{OUTER_RUN}_summary.json'; path.write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8'); print(json.dumps(output,indent=2,allow_nan=False))

if __name__=='__main__': main()
