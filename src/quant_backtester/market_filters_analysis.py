"""
Defines the various classes and implementation of filters used in the back tester.

In general, there are 5 classes of filters.
    - Pre-training filters to determine the assets universe.
        - _corr_filter
        - _market_cap_filter
        - _volatility_filter
        - _cointegration_filter
        - _mv_filter_cross_assets
    - Period filters to determine which portfolios are filtered
        - These are implemented indirectly in run()
        - Note this is the only filter which acts on portfolios themselves, and not assets within a portfolio
    - Market state filters to determine which strategies to use based on market state variables
        - _regime_estimator
    - Rolling filters to determine which assets to assign non-zero weights or to give ranked weights executed for
      defined frequency of time steps
        - The implementation is to define the direction of the biggest/smallest combination of the defined aspect and to apply the condition
          the chosen weight vector should be orthogonal to this direction
        - Neutrality conditions can also be implemented
        - _rolling_filter (highest z_score, highest sharpe, lowest standard deviation, highest mean, most positive mean count)
        - Note all are on returns of assets
    - Weight filters to help assign weights depending on the defined aspects of a portfolio
        - _pc_filter_weights (first n eigenvectors)
        - _beta_filter_weights
        - dollar neutrality is implemented in strategies._weights_alloc
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as m
from statsmodels.tsa.stattools import coint

from src.quant_backtester.data_filter import get_time_period, get_info

np.set_printoptions(threshold=None)

"""Defines the market assets by cap size. Standard def used."""
MARKET_CAP_BUCKETS = {
    'micro': (0.0, 300e6),
    'small': (300e6, 2e9),
    'mid': (2e9, 10e9),
    'large': (10e9, 200e9),
    'mega': (200e9, np.inf),
}


def classify_market_caps(market_caps):
    """Classify non-negative USD market caps into disjoint size buckets.

    The intervals are left-inclusive and right-exclusive, except ``mega``
    which has no upper bound. Missing, infinite, and negative observations are
    left unclassified rather than being silently assigned to a bucket.
    """
    caps = pd.to_numeric(pd.Series(market_caps, copy=True), errors='coerce')
    labels = pd.Series(pd.NA, index=caps.index, dtype='string', name='market_cap_bucket')
    valid = caps.notna() & np.isfinite(caps) & caps.ge(0.0)
    for label, (lower, upper) in MARKET_CAP_BUCKETS.items():
        labels.loc[valid & caps.ge(lower) & caps.lt(upper)] = label
    return labels


def market_cap_filter(strat_param, type_, market_caps=None):
    """
    Filter the requested asset universe using a point-in-time market-cap series.

    ``market_caps`` should be the information available at the selection time.
    When omitted, the bundled current metadata snapshot is used; that fallback
    is suitable for a current screen, not for a historical bias-free backtest.

    Args:
        strat_param: dict 
        type_: str
            Whether to return 'mega', 'small', 'mid', 'large' cap assets

    Returns: np.array
            Array of stocks which pass the given filter
    """

    if not isinstance(strat_param, dict) or not strat_param.get('stock_list'):
        raise ValueError("strat_param['stock_list'] must contain at least one asset")

    bucket = str(type_).lower()
    if bucket == 'medium':
        bucket = 'mid'
    if bucket not in MARKET_CAP_BUCKETS:
        allowed = ', '.join(MARKET_CAP_BUCKETS)
        raise ValueError(f"unknown market-cap bucket {type_!r}; expected one of: {allowed}")

    caps = get_info('marketCap') if market_caps is None else pd.Series(market_caps, copy=True)
    # Preserve caller order, remove duplicate symbols, and never introduce an
    # asset merely because it happens to exist in the metadata table.
    universe = pd.Index(dict.fromkeys(strat_param['stock_list']))
    caps = caps.reindex(universe)
    labels = classify_market_caps(caps)
    return pd.Index(labels.index[labels.eq(bucket)], name='asset')


def volatility_filter(strat_param, q_threshold) :
    """
    Volatility filter which returns all assets which fall below a given quantile
        Defining factor is average variance over the given period
    Note:

    Args:
        strat_param: dict 
        q_threshold: float
          What quantile of assets to return as classified by average variance

    Returns: np.array
        Array of stocks which pass the given filter
    """
    time_period = strat_param['time_period']
    base_data = get_time_period(strat_param['stock_list'], freq=strat_param['freq'], time_peri=time_period)
    var = base_data.var()
    var = var.rank(pct=True)
    var = var[var < var.quantile(q_threshold)].dropna()

    return np.array(var.index)


def cointegration_filter(strat_param, show_graphs=False)  :
    """
    Executes the Engel-Granger test on a pair of assets to determine if cointegrated on the given time period.

    Note:
        - Meant to use in tandem with runner_multiple
        - Meant for use for only one asset pair
        - Threshold is set at .05

    Args:
        strat_param: dict 
        show_graphs: bool, default is False
            Meant for visualization of the residual
            The rolling average residual is shown with a set roll of 20

    Returns: np.array
        Single element Boolean array

    """
    time_period = strat_param['time_period']
    cur_pair = get_time_period(strat_param['stock_list'], freq=strat_param['freq'], time_peri=time_period)
    cur_stock = strat_param['stock_list']

    if len(cur_stock) != 2:
        raise ValueError('cointegration_filter requires exactly two assets')
    if (cur_pair <= 0).any().any():
        return np.array([np.nan])
    log_pair = np.log(cur_pair)
    model = m.OLS(log_pair[cur_stock[0]], m.add_constant(log_pair[cur_stock[1]])).fit()
    p_value = coint(log_pair[cur_stock[0]], log_pair[cur_stock[1]])[1]

    if show_graphs:
        return model.resid.rolling(20).mean().vbt.plot(title=tuple(cur_stock[0:2]).__str__()).to_html(
            include_plotlyjs='cdn', include_mathjax=False, auto_play=False, full_html=False)
    return np.array([p_value])


def corr_filter(strat_param, c_threshold) :
    """

    Correlation filter which all assets which have a pairwise correlation less than the given threshold

    Notes
        - Pre-training filter
        - For assets that do have a higher correlation, only one of the assets is returned
    Args:
        strat_param: dict 
        c_threshold: float

    Returns: np.array
        Array of stocks which pass the given filter

    """
    time_period = strat_param['time_period']
    base_data = get_time_period(strat_param['stock_list'], freq=strat_param['freq'], time_peri=time_period)
    corr = base_data.corr()
    corr_lower = np.tril(corr, k=-1)
    corr = pd.DataFrame(corr_lower, index=corr.index, columns=corr.columns)
    corr = corr[corr.rank(pct=True) < c_threshold]
    return np.array(corr.index)


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """Causal percentile rank of the current value inside its trailing window."""
    return series.rolling(window, min_periods=window).rank(pct=True)


def _ewm_average_autocorrelation(market: pd.Series, half_life: float,
                                 lags=(1, 3, 5, 7, 9)) -> pd.Series:
    """Proper centered EWM Pearson autocorrelation averaged across causal lags."""
    correlations = []
    for lag in lags:
        lag = int(lag)
        if lag <= 0:
            raise ValueError('autocorrelation lags must be positive')
        correlations.append(
            market.ewm(halflife=half_life, adjust=False).corr(market.shift(lag)).clip(-1, 1)
        )
    return pd.concat(correlations, axis=1).mean(axis=1)


def _cap_regime_weights(weights: pd.DataFrame, maximum: float) -> pd.DataFrame:
    """Cap sleeve weights and redistribute excess without exceeding the cap."""
    maximum = float(maximum)
    if not 0 < maximum <= 1 or maximum * weights.shape[1] < 1:
        raise ValueError('max_sleeve_weight is infeasible')
    output = []
    for values in weights.to_numpy(dtype=float):
        values = np.maximum(values, 0.0)
        allocated = np.zeros_like(values)
        active = values > 0
        remaining = 1.0
        while active.any() and remaining > 1e-15:
            proposal = remaining * values[active] / values[active].sum()
            saturated = proposal > maximum
            indices = np.flatnonzero(active)
            if not saturated.any():
                allocated[indices] = proposal
                break
            saturated_indices = indices[saturated]
            allocated[saturated_indices] = maximum
            active[saturated_indices] = False
            remaining = 1.0 - allocated.sum()
        output.append(allocated)
    return pd.DataFrame(output, index=weights.index, columns=weights.columns)


def _regime_weights_from_returns(base_returns, asset_returns, roll, half_life,
                                 low_quantile=0.3, high_quantile=0.7,
                                 autocorrelation_deadband=0.05,
                                 allocation_half_life=5,
                                 rebalance_every=5,
                                 max_sleeve_weight=0.4):
    """Build causal strategy weights from already-computed return series."""
    roll = int(roll)
    half_life = float(half_life)
    low_quantile = float(low_quantile)
    high_quantile = float(high_quantile)
    if roll < 3:
        raise ValueError('regime roll must be at least 3')
    if not np.isfinite(half_life) or half_life <= 0:
        raise ValueError('regime half_life must be positive and finite')
    if not (0 < low_quantile < high_quantile < 1):
        raise ValueError('regime quantiles must satisfy 0 < low < high < 1')
    autocorrelation_deadband = float(autocorrelation_deadband)
    if not 0 <= autocorrelation_deadband < 0.25:
        raise ValueError('autocorrelation_deadband must satisfy 0 <= deadband < 0.25')
    allocation_half_life = float(allocation_half_life)
    rebalance_every = int(rebalance_every)
    if allocation_half_life <= 0 or rebalance_every <= 0:
        raise ValueError('allocation_half_life and rebalance_every must be positive')

    market = pd.Series(base_returns.squeeze(), dtype=float).replace([np.inf, -np.inf], np.nan)
    assets = pd.DataFrame(asset_returns, dtype=float).replace([np.inf, -np.inf], np.nan)
    common_index = market.index.intersection(assets.index)
    market = market.reindex(common_index)
    assets = assets.reindex(common_index)

    market_var = market.pow(2).ewm(halflife=half_life, adjust=False).mean()
    dispersion = assets.std(axis=1, ddof=1)

    # A full N-by-N EWM correlation matrix is quadratic in the universe size.
    # Standardized innovations provide a causal O(T*N) co-movement estimate:
    # the identity below is the mean product across all distinct asset pairs.
    asset_mean = assets.ewm(halflife=half_life, adjust=False).mean()
    asset_second_moment = assets.pow(2).ewm(halflife=half_life, adjust=False).mean()
    asset_variance = (asset_second_moment - asset_mean.pow(2)).clip(lower=0)
    standardized = (assets - asset_mean).div(asset_variance.pow(0.5).replace(0, np.nan))
    count = standardized.notna().sum(axis=1)
    pair_product = (
        standardized.sum(axis=1).pow(2) - standardized.pow(2).sum(axis=1)
    ).div((count * (count - 1)).where(count >= 2))
    average_correlation = pair_product.ewm(halflife=half_life, adjust=False).mean().clip(-1, 1)

    autocorrelation = _ewm_average_autocorrelation(market, half_life)

    features = pd.DataFrame({
        'var': _rolling_percentile(market_var, roll),
        'corr': _rolling_percentile(average_correlation, roll),
        'dis': _rolling_percentile(dispersion, roll),
        'ac': autocorrelation,
    }).dropna()

    scores = pd.DataFrame(0.05, index=features.index, columns=[
        'cross_asset_mv', 'mv', 'momentum_trending',
        'cross_asset_momentum_trending', 'cointegration'])
    # Continuous strengths avoid abrupt percentile-boundary switches.
    high_var = ((features['var'] - 0.5) / 0.5).clip(0, 1)
    low_var = ((0.5 - features['var']) / 0.5).clip(0, 1)
    middle_var = (1.0 - (features['var'] - 0.5).abs() / 0.5).clip(0, 1)
    high_corr = ((features['corr'] - 0.5) / 0.5).clip(0, 1)
    low_corr = ((0.5 - features['corr']) / 0.5).clip(0, 1)
    middle_corr = (1.0 - (features['corr'] - 0.5).abs() / 0.5).clip(0, 1)
    high_dispersion = ((features['dis'] - 0.5) / 0.5).clip(0, 1)
    positive_ac = ((features['ac'] - autocorrelation_deadband) /
                   (0.25 - autocorrelation_deadband)).clip(0, 1)
    negative_ac = ((-features['ac'] - autocorrelation_deadband) /
                   (0.25 - autocorrelation_deadband)).clip(0, 1)

    scores['cross_asset_mv'] += 1.2 * negative_ac + 0.8 * high_dispersion + 0.5 * high_var + 0.4 * low_corr
    scores['mv'] += 1.2 * negative_ac + 0.8 * low_var + 0.5 * low_corr
    scores['momentum_trending'] += 1.2 * positive_ac + 0.7 * high_corr
    scores['cross_asset_momentum_trending'] += 1.2 * positive_ac + 0.8 * high_dispersion
    scores['cointegration'] += 0.8 * negative_ac + 0.6 * middle_corr + 0.4 * middle_var + 0.6 * high_dispersion

    weights = scores.div(scores.sum(axis=1), axis=0)
    weights = _cap_regime_weights(weights, max_sleeve_weight)
    weights = weights.ewm(halflife=allocation_half_life, adjust=False).mean()
    weights = _cap_regime_weights(weights.div(weights.sum(axis=1), axis=0), max_sleeve_weight)
    update_rows = np.arange(len(weights)) % rebalance_every == 0
    weights.loc[~update_rows] = np.nan
    return weights.ffill().dropna()


def regime_estimator(strat_param) -> pd.DataFrame:
    """
    Helper to determine what strategy to use depending on the regime

    Note:
        - Uses four causal state variables implemented with rolling quantities:
          market variance, cross-sectional dispersion, average correlation,
          and signed market autocorrelation.

    Args:
        strat_param: dict 

    Returns: pd.DataFrame

    """

    time_period = strat_param['time_period']
    path = Path(__file__).parents[2]
    all_assets = pd.read_parquet(path / 'data/close_1d_10y.parquet').columns
    stck_data = get_time_period(all_assets, freq=strat_param['freq'], time_peri=time_period)
    stck_data = stck_data.drop(columns='SPY', errors='ignore').pct_change()
    base_data = get_time_period(['SPY'], freq=strat_param['freq'], time_peri=time_period).pct_change()
    parameters = strat_param['weights_filter']['regime_estimator']
    return _regime_weights_from_returns(
        base_data, stck_data,
        roll=parameters['roll'],
        half_life=parameters['half_life'],
        low_quantile=parameters.get('low_quantile', 0.3),
        high_quantile=parameters.get('high_quantile', 0.7),
        autocorrelation_deadband=parameters.get('autocorrelation_deadband', 0.05),
        allocation_half_life=parameters.get('allocation_half_life', 5),
        rebalance_every=parameters.get('rebalance_every', 5),
        max_sleeve_weight=parameters.get('max_sleeve_weight', 0.4),
    )

def _rolling_filter(strat_param,  filters_r=None) -> np.ndarray:
    """
    Applies rolling filters as defined above.

    Args:
        strat_param: dict 
            Strategy parameters, all of which are mayhaps not necessary to use for the given list of filters.
        filters_r:
            List of filters as keys with any associated parameters as the values.

    Returns:
        Returns a boolean dataframe with the same axes as the signals dataframe. Indicates whether to include the asset,
        which are listed as the columns.

    """

    time_period = strat_param['time_period']

    roll = filters_r['roll']
    if 'type' in filters_r.keys():
        type_ = filters_r['type']
    else:
        type_ = 'orthongonal_direction_neutrality'
    assets  = strat_param['stock_list']

    stck_data = get_time_period(list(dict.fromkeys(assets)) , freq=strat_param['freq'],
                                time_peri=time_period)

    stck_data = stck_data.pct_change()
    metrics = filters_r['metrics']
    cond_length = len(metrics)
    if 'sector' in metrics  :
        sectors = get_info(['sectorKey']).loc[assets].groupby('sectorKey').groups
        cond_length += len( sectors.keys   ())-1
    if not metrics:
        metrics = ['mean']
    cond = np.zeros((len(stck_data.index),  cond_length ,len(stck_data.columns)) )
    index = 0


    if type_ == 'neutrality':
        for metric in metrics:
            if metric == 'value':
                raw_values = get_info(['priceToBook','trailingPE','enterpriseToEbitda'])
                z_score = (raw_values -raw_values.mean())/raw_values.std()
                z_score = z_score.T[assets].mean()

                cond[:, index, :] = np.broadcast_to(z_score.T.values,(cond.shape[0],len(z_score)))
                index += 1
            elif metric == 'sector':

                for sector_key,asset_list in  sectors .items():
                    dt = pd.DataFrame(0,columns= assets,index = np.zeros(cond.shape[0]))
                    dt[asset_list] = 1

                    cond[:,index,:]  =dt.values
                    index += 1

            index += 1



    norm = sum(list(range(1,len(stck_data.columns) + 1)))
    for type_ in metrics:
        if type_ == 'z_score':
            std = stck_data.rolling(roll).std()
            mean = stck_data.rolling(roll).mean()
            z_score = (stck_data - mean) / std
            filter_ser = z_score
            filter_ser =  (filter_ser.values.argsort(axis=1)[::-1]).argsort(axis=1) + 1
        elif type_ == 'std':
            filter_ser = stck_data.rolling(roll).std()
            filter_ser =  (filter_ser.values.argsort(axis=1)).argsort(axis=1) + 1

        elif type_ == 'mean':
            filter_ser = stck_data.rolling(roll).mean()
            filter_ser =  (filter_ser.values.argsort(axis=1)[::-1]).argsort(axis=1) + 1

        elif type_ == 'mean_inc_v_d':
            mean = stck_data.rolling(roll).mean()
            filter_ser = mean.where(mean > 0, axis=1).count(axis=1)[::-1 ]   .argsort(axis=1) + 1
        elif type_ == 'sharpe':
            filter_ser = stck_data.rolling(roll).mean()/stck_data.rolling(roll).std()
            filter_ser =  (filter_ser.values.argsort(axis=1)[::-1]).argsort(axis=1) + 1

        filter_ser = filter_ser / norm
        cond[:, index, :] = filter_ser
        index+=1
        # else:
        #     filter_ser = stck_data.rolling(roll).mean().dropna() / stck_data.rolling(roll).std().dropna()

    # ranked_assets = np.array([[False] * len(filter_ser.columns)] * len(filter_ser)) + 0
    # top_k = np.array([len(filter_ser.columns), n]).min()
    #
    #     indx = np.argpartition(filter_ser.values, top_k, axis=1)
    #
    #     np.put_along_axis(ranked_assets,
    #                       indx[:, :top_k], True, 1)


    return cond


def _beta_filter_weights(strat_param) -> pd.DataFrame:
    """
    Returns a rolling collection of beta values for each given asset, as related to the baseline, over
    the given time period.

    Notes:
        - Weights filter.
    Args:
        strat_param: dict 

    Returns: pd.DataFrame

    """
    time_period = strat_param['time_period']
    assets = list(dict.fromkeys(strat_param['stock_list']))
    prices = get_time_period(assets + ['SPY'], freq=strat_param['freq'], time_peri=time_period)
    returns = prices.pct_change()
    roll = int(strat_param['weights_filter']['beta']['roll'])
    if roll < 2:
        raise ValueError('beta roll must be at least 2')
    market = returns['SPY']
    market_variance = market.rolling(roll, min_periods=roll).var().replace(0.0, np.nan)
    beta = returns[assets].rolling(roll, min_periods=roll).cov(market).div(market_variance, axis=0)
    return beta.replace([np.inf, -np.inf], np.nan).dropna(how='all').reindex(columns=assets)


def _pc_filter_weights(strat_param, ) -> np.ndarray:
    """
    Returns the n top eigenvectors of the correlation matrix over the given time period, with the given
    roll.

    Notes:
        - Weights filter.
    Args:
        strat_param: dict 

    Returns: np.ndarray

    """

    time_period = strat_param['time_period']
    roll = strat_param['weights_filter']['pc']['roll']
    n = strat_param['weights_filter']['pc']['n']

    assets = list(dict.fromkeys(strat_param['stock_list']))
    if roll < 2 or n < 1 or n >= len(assets):
        raise ValueError('PC settings require roll >= 2 and 1 <= n < number of assets')
    returns = get_time_period(assets, freq=strat_param['freq'], time_peri=time_period).pct_change()
    values = returns.to_numpy(dtype=float)
    constraints = []

    for end in range(roll, len(values)):
        window = values[end - roll + 1:end + 1]
        if not np.isfinite(window).all():
            continue
        # window_= window[:,~(window == 0).all(axis=0)]
        with np.errstate(divide="ignore", invalid="ignore"):
            correlation = np.corrcoef(window, rowvar=False)

        if not np.isfinite(correlation).all():
            continue

        _, eigenvectors = np.linalg.eigh(correlation)
        # eigh returns eigenvectors in columns and ascending eigenvalue order.
        # Constraint rows must therefore be the final n columns transposed.

        constraints.append(eigenvectors[:, -n:].T)
    if not constraints:
        return np.empty((0, n, len(assets)))
    return np.stack(constraints)


def mv_filter_cross_assets(strat_param) -> pd.DataFrame:
    """

    Args:
        strat_param: dict

    Returns: pd.DataFrame

    """
    time_period = strat_param['time_period']
    roll = strat_param['parameters_']['roll']

    stck_data = get_time_period(strat_param['stock_list'], freq=strat_param['freq'], time_peri=time_period)
    stck_data = stck_data.pct_change()
    stck_data_cr = stck_data.T

    std = stck_data_cr.std()
    mean = stck_data_cr.mean()
    z_score = (stck_data_cr - mean) / std

    z_score_end = z_score.T.mean()

    return z_score_end


def get_analysis(results=None, **kwargs) -> None:
    """
    Generates HTML document of analysis

    Args:
        results: list, default is None 

    Returns: None 

    """

    track = []
    html_ = [
        '<html style="display: flex; justify-content: center"><body style="background-color:#1a1a1a ;  color:white">']

    results = [list(x) for x in results]
    if not results: return
    for x in results:
        for y in kwargs['parameters']:
            kwargs['time_period'] = y
            kwargs['stock_list'] = x

            graph = kwargs['filter_func'](
                kwargs, )
        if graph:
            html_.append('<div style="display: flex">'), html_.append(graph[0]), html_.append(graph[1])
            html_.append('</div>'),
            html_.append(graph[2]),
            html_.append(graph[3]),
            html_.append('<br><br<br><br>')
            track.append(dict(stock=x, elem=html_[-7:], metric=graph[-1]))
    track.sort(key=lambda x: x['metric'])

    html_ = [y for x in track for y in x['elem']]
    html_ = ['<html style="display: flex; justify-content: center"><body style="background-color:#1a1a1a ;  color:white">'] + html_
    html_.append('</body></html>'),
    path = Path(__file__).parents[2]

    path_name = str(path) + '/docs/results/'+ str(kwargs['time_period']) + str(kwargs['strat_class'].keys()) + '.html'
    Path(path_name).write_text('\n'.join(html_), encoding='utf-8')
