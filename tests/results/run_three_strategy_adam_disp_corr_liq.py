"""Adam-optimized preset correlation/liquidity/dispersion allocation."""
import json, os, sys
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

os.environ.setdefault('NUMBA_DISABLE_JIT', '1')
ROOT = Path (__file__).resolve().parent.parent.parent
PROJECT = ROOT
sys.path[:0] = [str(PROJECT/'src/quant_backtester'), str(ROOT/'artifacts')]

from src.quant_backtester import get_time_period
from src.quant_backtester.hmm_regime import build_hmm_features
from src.quant_backtester.strategies import (_get_signals_mv, _get_signals_mv_cross_asset,
                 _get_signals_momentum_tr, _get_signals_momentum_cross_asset)
from tests.run_cmv_full_three_stage_five_cycles import CYCLES, FEE, SLIPPAGE, performance
from tests.run_cross_momentum_timeseries_momentum_rule_regime import normalize, net_returns, passed
from tests.run_cmv_mt_cmt_rule_regime import build_correlation_liquidity_dispersion_features
from tests.run_three_strategy_preset_regimes import allocations as disp_allocations, combine, SLEEVES
from tests. run_cmv_mt_cmt_bayesian_hmm_regime import select_strategy_on_validation

REBALANCES = (5, 10, 20)
ITERATIONS = 100
RESTARTS = 2
MODE = os.environ.get('ALLOCATION_OPTIMIZER', 'adam').lower()
FEATURE_SET = os.environ.get('PRESET_FEATURE_SET', 'disp_corr_liq').lower()
ROLLING_FIXED = os.environ.get('ROLLING_FIXED_WINDOWS','false').lower()=='true'
INCLUDE_TIME_MV = os.environ.get('INCLUDE_TIME_MV', 'false').lower() == 'true'
if INCLUDE_TIME_MV: SLEEVES=(*SLEEVES,'mean_reversion')
USE_NEUTRALITY = os.environ.get('USE_NEUTRALITY', 'false').lower() == 'true'
NEUTRALITY_MODE = os.environ.get('NEUTRALITY_MODE', 'all').lower()
NEUTRAL_ROLL = 60
if MODE not in {'adam','bayesian','grid'}:
    raise ValueError(f'ALLOCATION_OPTIMIZER must be adam, bayesian, or grid; got {MODE!r}')
if FEATURE_SET not in {'disp_corr_liq','ac_corr_vol'}:
    raise ValueError(f'PRESET_FEATURE_SET must be disp_corr_liq or ac_corr_vol; got {FEATURE_SET!r}')
if NEUTRALITY_MODE not in {'all','beta','beta_parallel','dollar','pc'}:
    raise ValueError(
        'NEUTRALITY_MODE must be all, beta, beta_parallel, dollar, or pc; '
        f'got {NEUTRALITY_MODE!r}'
    )

def neutrality_cache(columns, returns, market):
    r=returns[columns]; market_var=market.rolling(NEUTRAL_ROLL,min_periods=NEUTRAL_ROLL).var()
    beta=r.rolling(NEUTRAL_ROLL,min_periods=NEUTRAL_ROLL).cov(market).div(market_var,axis=0).to_numpy()
    projection=np.zeros((len(r),len(columns),len(columns))); pc=np.full((len(r),len(columns)),np.nan)
    values=r.to_numpy()
    for t in range(NEUTRAL_ROLL-1,len(r)):
        corr=np.corrcoef(values[t-NEUTRAL_ROLL+1:t+1],rowvar=False)
        if not np.isfinite(corr).all() or not np.isfinite(beta[t]).all(): continue
        _,vectors=np.linalg.eigh(corr); pc[t]=vectors[:,-1]
        if NEUTRALITY_MODE == 'beta_parallel':
            b=beta[t]; projection[t]=np.outer(b,b)/max(float(b@b),1e-12); continue
        if NEUTRALITY_MODE == 'beta': c=np.vstack((beta[t],))
        elif NEUTRALITY_MODE == 'dollar': c=np.vstack((np.ones(len(columns)),))
        elif NEUTRALITY_MODE == 'pc': c=np.vstack((pc[t],))
        else: c=np.vstack((np.ones(len(columns)),beta[t],pc[t]))
        projection[t]=np.eye(len(columns))-c.T@np.linalg.pinv(c@c.T)@c
    return {'columns':columns,'projection':projection,'beta':beta,'pc':pc}

def neutralize(target, cache):
    frame=target.reindex(columns=cache['columns']).fillna(0.); raw=frame.to_numpy()
    projected=np.einsum('ti,tij->tj',raw,cache['projection'])
    original_gross=np.abs(raw).sum(axis=1); projected_gross=np.abs(projected).sum(axis=1)
    active=projected_gross>1e-12; projected[active]*=(original_gross[active]/projected_gross[active])[:,None]
    projected[~active]=0.; return pd.DataFrame(projected,index=frame.index,columns=frame.columns)

def neutrality_residuals(weights, cache, index):
    loc=weights.index.get_indexer(index); w=weights.reindex(index).fillna(0.).to_numpy(); active=np.abs(w).sum(axis=1)>1e-12
    def stats(values):
        finite=np.isfinite(values)&active; return {'mean_abs':float(np.mean(np.abs(values[finite]))) if finite.any() else 0.,'max_abs':float(np.max(np.abs(values[finite]))) if finite.any() else 0.}
    result={'dollar':stats(w.sum(axis=1)),'beta':stats(np.nansum(w*cache['beta'][loc],axis=1)),'leading_pc':stats(np.nansum(w*cache['pc'][loc],axis=1))}
    if NEUTRALITY_MODE=='beta_parallel':
        projected=np.einsum('ti,tij->tj',w,cache['projection'][loc]); denominator=np.linalg.norm(w,axis=1)
        error=np.divide(np.linalg.norm(w-projected,axis=1),denominator,out=np.zeros_like(denominator),where=denominator>1e-12)
        result['beta_parallel_relative_error']=stats(error)
    return result

def allocations(features, half, rebalance, cmv_scale, mt_scale, cmt_scale, cash_scale, mv_scale=None):
    if FEATURE_SET != 'ac_corr_vol':
        if mv_scale is None: return disp_allocations(features, half, rebalance, cmv_scale, mt_scale, cmt_scale, cash_scale)
        f=features.reindex(columns=['corr','liq','dis']).dropna(); high_corr=((f['corr']-.5)/.5).clip(0,1); low_corr=((.5-f['corr'])/.5).clip(0,1); high_liq=((f['liq']-.5)/.5).clip(0,1); low_liq=((.5-f['liq'])/.5).clip(0,1); high_dis=((f['dis']-.5)/.5).clip(0,1); low_dis=((.5-f['dis'])/.5).clip(0,1)
        score=pd.DataFrame(index=f.index); score['cross_asset_mv']=(.05+high_dis+low_corr+.3*(1-high_liq))*cmv_scale; score['momentum_trending']=(.05+high_corr+low_dis+.6*high_liq)*mt_scale; score['cross_asset_momentum_trending']=(.05+high_dis+low_corr+.8*high_liq)*cmt_scale; score['mean_reversion']=(.05+high_dis+low_corr+.5*low_liq)*mv_scale; score['cash']=(.05+1.5*low_liq+.3*high_corr*high_dis)*cash_scale
        weight=score.div(score.sum(axis=1),axis=0).ewm(halflife=half,adjust=False).mean(); weight.loc[np.arange(len(weight))%rebalance!=0]=np.nan; return weight.ffill().fillna(0.)
    f=features.reindex(columns=['ac','var','corr']).dropna(); ac=f.ac.clip(-1,1)
    positive=ac.clip(lower=0); negative=(-ac).clip(lower=0)
    high_var=((f['var']-.5)/.5).clip(0,1)
    high_corr=((f['corr']-.5)/.5).clip(0,1); low_corr=((.5-f['corr'])/.5).clip(0,1)
    score=pd.DataFrame(index=f.index)
    score['cross_asset_mv']=(.05+2*negative+.6*low_corr)*cmv_scale
    score['momentum_trending']=(.05+2*positive+.8*high_corr)*mt_scale
    score['cross_asset_momentum_trending']=(.05+2*positive+.8*low_corr)*cmt_scale
    score['cash']=(.05+1.2*high_var+.5*(1-ac.abs().clip(0,1)))*cash_scale
    weight=score.div(score.sum(axis=1),axis=0).ewm(halflife=half,adjust=False).mean()
    weight.loc[np.arange(len(weight))%rebalance!=0]=np.nan
    return weight.ffill().fillna(0.)

def decode(theta):
    sig = 1 / (1 + np.exp(-np.clip(theta, -20, 20)))
    half = 1.0 + 19.0 * sig[0]
    scales = np.exp(np.log(.25) + (np.log(4.0)-np.log(.25))*sig[1:])
    return (half, *scales)

def adam_spsa(objective, seed):
    """Adam using deterministic SPSA gradient estimates (two backtests per step)."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, .35, 5); m = np.zeros(5); v = np.zeros(5)
    best = (objective(theta), theta.copy())
    for t in range(1, ITERATIONS + 1):
        delta = rng.choice((-1.0, 1.0), 5)
        c = .12 / (t ** .101)
        grad = (objective(theta + c*delta)-objective(theta-c*delta))/(2*c) * delta
        m = .9*m + .1*grad; v = .999*v + .001*grad*grad
        step = .06 * np.sqrt(1-.999**t)/(1-.9**t)
        theta += step*m/(np.sqrt(v)+1e-8)
        value = objective(theta)
        if value > best[0]: best = (value, theta.copy())
    return best

def bayesian_optimize(objective, seed, initial=12, iterations=28, dimensions=5):
    """Gaussian-process Bayesian optimization with expected improvement."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.5, 2.5, size=(initial, dimensions))
    y = np.array([objective(row) for row in x])
    kernel = Matern(length_scale=np.ones(dimensions), nu=2.5) + WhiteKernel(noise_level=1e-5)
    for _ in range(iterations):
        gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True,
                                      random_state=seed).fit(x, y)
        candidates = rng.uniform(-3.5, 3.5, size=(1500, dimensions))
        mu, sigma = gp.predict(candidates, return_std=True)
        improvement = mu - y.max() - .01
        z = np.divide(improvement, sigma, out=np.zeros_like(improvement), where=sigma > 1e-12)
        ei = improvement*norm.cdf(z) + sigma*norm.pdf(z)
        nxt = candidates[int(np.argmax(ei))]
        x = np.vstack((x, nxt)); y = np.append(y, objective(nxt))
    index = int(np.argmax(y))
    return float(y[index]), x[index].copy()

def main():
    source = json.loads((ROOT/'artifacts'/('checkpoint_rolling_fixed_500_260_260_three_strategy_corr_liq_disp_rbf_svm_summary.json' if ROLLING_FIXED else 'checkpoint_three_strategy_parameter_optimized_ac_vol_corr_hmm_summary.json')).read_text())
    universe = pd.read_parquet(PROJECT/'data'/'close_1d_10y.parquet').columns.tolist()
    prices = get_time_period(universe, time_peri=(0,2060)); returns = prices.pct_change().fillna(0.)
    market = get_time_period(['SPY'], time_peri=(0,2060)).reindex(prices.index)['SPY'].pct_change().fillna(0.)
    volume = pd.read_parquet(PROJECT/'data'/'volume_1d_10y.parquet', columns=universe).reindex(prices.index)
    features = (build_hmm_features(market, returns).loc[:,['ac','var','corr']]
                if FEATURE_SET == 'ac_corr_vol' else
                build_correlation_liquidity_dispersion_features(prices, volume, returns))
    cache = {}; runs = []
    def raw(name, p):
        key=(name,tuple(sorted(p.items())))
        if key not in cache:
            cfg={'stock_list':universe,'time_period':(0,2060),'freq':'d','strat_class':{name:p},'parameters_':p}
            cache[key]=(_get_signals_mv_cross_asset(cfg) if name=='cross_asset_mv' else
                (_get_signals_momentum_tr(cfg,prices) if name=='momentum_trending' else
                 (_get_signals_momentum_cross_asset(cfg) if name=='cross_asset_momentum_trending' else _get_signals_mv(cfg,prices)))).reindex(prices.index).fillna(0.)
        return cache[key]
    mv_candidates=[(p,raw('mean_reversion',p)) for p in ({'z_threshold':z,'roll':roll} for roll in (20,30,60) for z in (1.5,2.,2.5))] if INCLUDE_TIME_MV else []
    cycles=([{'training':(260*i,500+260*i),'validation':(500+260*i,760+260*i),'held_out':(760+260*i,1020+260*i)} for i in range(5)] if ROLLING_FIXED else CYCLES)
    for number,(periods,prior) in enumerate(zip(cycles,source['runs']),1):
        selected_strategies=dict(prior['selected_strategies'])
        if INCLUDE_TIME_MV:
            selected_strategies['mean_reversion'],_=select_strategy_on_validation('mean_reversion',mv_candidates,returns,market,prices.index,periods)
        sleeves={n:normalize(raw(n,selected_strategies[n]['parameters']),selected_strategies[n]['assets']) for n in SLEEVES}
        target_columns=list(dict.fromkeys(c for f in sleeves.values() for c in f.columns))
        neutral_cache=neutrality_cache(target_columns,returns,market) if USE_NEUTRALITY else None
        def target_for(alloc):
            target=combine(sleeves,alloc,prices.index)
            return neutralize(target,neutral_cache) if USE_NEUTRALITY else target
        val=prices.index[slice(*periods['validation'])]
        def score(theta, rebalance):
            decoded=decode(theta); half,cmv_s,mt_s,cmt_s,cash_s=decoded[:5]; mv_s=decoded[5] if INCLUDE_TIME_MV else None
            alloc=allocations(features,half,rebalance,cmv_s,mt_s,cmt_s,cash_s,mv_s)
            metric=performance(net_returns(target_for(alloc),returns).reindex(val).fillna(0.),market.reindex(val))
            return float(metric['sharpe']) if np.isfinite(metric['sharpe']) else -1e6
        choices=[]
        for rebalance in REBALANCES:
            if MODE == 'grid':
                for half,cmv_s,mt_s,cmt_s,cash_s in product((2.,5.,10.),(.5,1.,2.),(.5,1.,2.),(.5,1.,2.),(.5,1.,2.)):
                    alloc=allocations(features,half,rebalance,cmv_s,mt_s,cmt_s,cash_s)
                    metric=performance(net_returns(target_for(alloc),returns).reindex(val).fillna(0.),market.reindex(val))
                    choices.append({'smoothing_half_life':half,'rebalance_every_bars':rebalance,'cmv_scale':cmv_s,'mt_scale':mt_s,'cmt_scale':cmt_s,'cash_scale':cash_s,'validation':metric})
                continue
            repeats = RESTARTS if MODE == 'adam' else 1
            for restart in range(repeats):
                optimizer = adam_spsa if MODE == 'adam' else bayesian_optimize
                value,theta=(optimizer(lambda x:score(x,rebalance),1000*number+100*rebalance+restart,dimensions=6) if MODE=='bayesian' and INCLUDE_TIME_MV else optimizer(lambda x:score(x,rebalance),1000*number+100*rebalance+restart))
                decoded=decode(theta); half,cmv_s,mt_s,cmt_s,cash_s=decoded[:5]; mv_s=decoded[5] if INCLUDE_TIME_MV else None
                alloc=allocations(features,half,rebalance,cmv_s,mt_s,cmt_s,cash_s,mv_s)
                metric=performance(net_returns(target_for(alloc),returns).reindex(val).fillna(0.),market.reindex(val))
                choices.append({'smoothing_half_life':half,'rebalance_every_bars':rebalance,'cmv_scale':cmv_s,'mt_scale':mt_s,'cmt_scale':cmt_s,'cash_scale':cash_s,**({'mv_scale':mv_s} if INCLUDE_TIME_MV else {}),'validation':metric})
        choices.sort(key=lambda x:(passed(x['validation']),x['validation']['sharpe'],x['validation']['total_return']),reverse=True)
        selected=choices[0]
        alloc=allocations(features,selected['smoothing_half_life'],selected['rebalance_every_bars'],selected['cmv_scale'],selected['mt_scale'],selected['cmt_scale'],selected['cash_scale'],selected.get('mv_scale'))
        held=prices.index[slice(*periods['held_out'])]
        final_target=target_for(alloc); metric=performance(net_returns(final_target,returns).reindex(held).fillna(0.),market.reindex(held))
        runs.append({'run':number,'selected_strategies':selected_strategies,'selected':selected,'average_held_out_allocations':alloc.reindex(held).mean().to_dict(),'neutrality_residuals':neutrality_residuals(final_target,neutral_cache,held) if USE_NEUTRALITY else None,'held_out':metric,'held_out_passed':passed(metric)})
        print(f'cycle {number}/5 complete',flush=True)
    names=('total_return','sharpe','alpha','max_drawdown')
    optimizer_name = 'Adam with SPSA validation gradients' if MODE == 'adam' else 'Gaussian-process Bayesian optimization with expected improvement'
    feature_names = ['autocorrelation','correlation','volatility'] if FEATURE_SET=='ac_corr_vol' else ['dispersion','correlation','liquidity']
    neutrality_names=({'beta':['rolling_beta'],'beta_parallel':['rolling_beta_parallel'],'dollar':['dollar'],'pc':['leading_pc']}.get(NEUTRALITY_MODE,['dollar','rolling_beta','leading_pc']) if USE_NEUTRALITY else [])
    strategy_label='CMV + MT + CMT + time-series MV' if INCLUDE_TIME_MV else 'CMV + MT + CMT'
    output={'test':f'{MODE.title()}-optimized {strategy_label} with preset regimes','features':feature_names,'optimizer':optimizer_name,'neutrality':neutrality_names,'neutrality_lookback_bars':NEUTRAL_ROLL if USE_NEUTRALITY else None,'adam_iterations':ITERATIONS if MODE=='adam' else None,'bayesian_evaluations_per_rebalance':40 if MODE=='bayesian' else None,'restarts_per_rebalance':RESTARTS if MODE=='adam' else 1,'discrete_rebalance_candidates':list(REBALANCES),'classifier':None,'execution':{'execution_delay_bars':1,'fee_per_order':FEE,'slippage_per_order':SLIPPAGE},'runs':runs,'average_held_out_metrics':{n:float(np.mean([r['held_out'][n] for r in runs])) for n in names},'held_out_pass_count':sum(r['held_out_passed'] for r in runs),'scientific_status':'Diagnostic: these historical windows were viewed earlier.'}
    if MODE=='bayesian' and FEATURE_SET=='ac_corr_vol': filename='checkpoint_three_strategy_bayesian_optimization_ac_corr_vol_preset_regimes_summary.json'
    elif MODE=='bayesian' and USE_NEUTRALITY and NEUTRALITY_MODE=='beta': filename='checkpoint_three_strategy_bayesian_disp_corr_liq_preset_beta_only_neutral_summary.json'
    elif MODE=='bayesian' and USE_NEUTRALITY and NEUTRALITY_MODE=='beta_parallel': filename='checkpoint_three_strategy_bayesian_disp_corr_liq_preset_beta_parallel_summary.json'
    elif MODE=='bayesian' and USE_NEUTRALITY: filename='checkpoint_three_strategy_bayesian_disp_corr_liq_preset_beta_dollar_pc_neutral_summary.json'
    elif MODE=='bayesian' and INCLUDE_TIME_MV: filename='checkpoint_four_strategy_bayesian_disp_corr_liq_preset_with_time_mv_summary.json'
    elif MODE=='bayesian': filename='checkpoint_three_strategy_bayesian_optimization_disp_corr_liq_preset_regimes_summary.json'
    elif FEATURE_SET=='ac_corr_vol': filename='checkpoint_three_strategy_adam_ac_corr_vol_preset_regimes_summary.json'
    else: filename='checkpoint_three_strategy_adam_disp_corr_liq_preset_regimes_summary.json'
    if ROLLING_FIXED:
        filename=f'checkpoint_rolling_fixed_500_260_260_three_strategy_{MODE}_{FEATURE_SET}_preset_regimes_summary.json'
    path=ROOT/'artifacts'/filename
    path.write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8'); print(json.dumps(output,indent=2,allow_nan=False))

if __name__=='__main__': main()
