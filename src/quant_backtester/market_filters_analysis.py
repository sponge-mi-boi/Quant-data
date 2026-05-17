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

import duckdb
import numpy as np
import pandas as pd
import plotly
import statsmodels.api as m
from statsmodels.tsa.stattools import coint

from quant_backtester.data_filter import get_time_period, get_info

np.set_printoptions(threshold=None)


def market_cap_filter(strat_param, type_)  :
    """
    Market cap filter to classify assets based on average market cap over the given time period
        Pre defined size thresholds

    Args:
        strat_param: dict 
        type_: str
            Whether to return 'mega', 'small', 'mid', 'large' cap assets

    Returns: np.array
            Array of stocks which pass the given filter
    """

    market_cap = get_info('marketCap')

    m_mega_threshold = 200 * 10** 9
    m_large_threshold = 10 * 10 ** 9
    m_small_threshold = 2 * 10 ** 9
    m_medium_threshold = 300 * 10 ** 6

    if type_ == 'mega':
        return market_cap[market_cap >= m_mega_threshold].index
    if type_ == 'large':
        return market_cap[(market_cap >= m_large_threshold) & (market_cap < m_mega_threshold)] .index
    elif type_ == 'medium':
        return market_cap[ (m_medium_threshold <= market_cap) & (market_cap < m_large_threshold) ] .index
    elif type_ == 'small':
        return market_cap [   (m_small_threshold <= market_cap) & (m_medium_threshold > market_cap ) ].dropna().index
    return None


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

    model = m.OLS((cur_pair[cur_stock[0]]), m.add_constant((cur_pair[cur_stock[1]]))).fit()
    results = coint(np.log(cur_pair[cur_stock[0]]), np.log(cur_pair[cur_stock[1]]))[1]

    if show_graphs:
        return model.resid.rolling(20).mean().vbt.plot(title=tuple(cur_stock[0:2]).__str__()).to_html(
            include_plotlyjs='cdn', include_mathjax=False, auto_play=False, full_html=False)
    arr = np.array([False])

    if results < .05 + .001:
        arr = np.array([True])
    return arr


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


def regime_estimator(strat_param) -> pd.DataFrame:
    """
    Helper to determine what strategy to use depending on the regime

    Note:
        - Supports 5 different states, defined as state variables, implemented with rolling quantities.
        - Market_state(roll) = (Market volatility, Dispersion, Correlation, autocorrelation, trend)(roll)

    Args:
        strat_param: dict 

    Returns: pd.DataFrame

    """

    time_period = strat_param['time_period']

    path = Path(__file__).parents[2]
    path = str(path)

    stck_data = get_time_period(pd.read_parquet(path + '/data/processed/close_1d_10y.parquet').columns,
                                freq=strat_param['freq'],
                                time_peri=time_period).drop(columns='SPY')
    base_data = get_time_period(['SPY'], freq=strat_param['freq'], time_peri=strat_param['time_period'])
    base_data = base_data.pct_change()

    stck_data = stck_data.pct_change()
    base_data = base_data.dropna()
    stck_data = stck_data.dropna()
    parameters = strat_param['weights_filter']['regime_estimator']
    name = path + '/data/processed/' + str (time_period ) + '.parquet'

    if Path(name).exists(): return pd.read_parquet(name)

    roll = parameters['roll']
    var_b = base_data.pow(2).ewm(halflife=parameters['half_life'], adjust=False).mean()

    var = var_b.rolling(roll).rank(pct=True)
    corr_stck_data = stck_data.ewm(halflife=parameters['half_life'],adjust=False).corr()


    corr = corr_stck_data.values.reshape(
        (len(var.index), len(stck_data.columns), len(stck_data.columns)))

    corr_sum =  (corr.sum(axis=-1 - 1).sum(axis=-1) - len(stck_data.columns)) / (corr.shape[--1]*(corr.shape[-1]-1))


    corr_ind = pd.Series(corr_sum, index=var.index) . dropna()
    corr_ind = corr_ind.rolling(roll).rank(pct=True)

    base_mean = base_data.ewm(halflife=parameters['half_life'], adjust=False).mean()
    std = var_b.pow(1/2)
    trend_val = base_mean/std

    trend = trend_val  .rolling(roll).rank(pct=True)

    disp_v = stck_data.std(axis = 1)
    disp = disp_v .rolling(roll).rank(pct= True)

    ac = pd.DataFrame(index=base_data.index)
    for x in range(1,10,2):
        ac[x] =( base_data*base_data.shift(x)).ewm(halflife=parameters['half_life'], adjust=False).mean()
    ac = ac.dropna().mean(axis=1)
    ac = ac.rolling(roll).rank(pct=True)
    vec = pd.concat([var , corr_ind,trend,disp,ac], axis=1) .dropna()

    vec.columns = ['var','corr','trend','dis','ac']

    vec_states = pd.DataFrame(data=0,index=vec.index,columns=np.array([0,1,2,3,4]))
    vec_states.loc[vec[vec['trend'] > .7].index,2] += 1
    vec_states.loc[vec[vec['corr'] > .7].index,2] += 1
    vec_states.loc[vec[vec['dis'] > .7].index,[0,3,4]] += 1
    vec_states.loc[vec[vec['var'] > .7].index,0] += 1
    vec_states.loc[vec[vec['ac'] < 0].index,[0,1,4]] += 1
    vec_states.loc[vec[vec['ac'] > 0].index,[ 2  ,3 ]] += 1
    vec_states.loc[vec[vec['corr'] < .3].index, [0,1]] += 1
    vec_states.loc[vec[vec['trend'] <  .3 ].index,1] += 1
    vec_states.loc[vec[vec['var'] < .3  ].index,1] += 1
    vec_states.loc[vec[( vec['corr'] > .3) & (vec['corr'] < .7)].index, [4]] += 1
    vec_states.loc[vec[  (vec['var'] > .3) & (vec['var']<.7) ].index, [4]] += 1
    vec_states = vec_states.div(vec_states.sum(axis=1), axis=0)

    columns = dict(cross_asset_mv=0,mv=1,momentum_trending=2,cross_asset_momentum_trending =3, cointegration=4 )
    vec_states.columns = columns.keys()

    vec_states.to_parquet(name)

    return vec_states

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
    stck_data = get_time_period(list(dict.fromkeys(strat_param['stock_list'])) + ['SPY'], freq=strat_param['freq'],
                                time_peri=time_period)
    stck_data = stck_data.pct_change().dropna()
    roll = strat_param['weights_filter']['beta']['roll']

    cov = stck_data.rolling(roll).cov()['SPY']
    var = stck_data.rolling(roll).var()
    val = cov.values.reshape(len(cov.index.levels[1]), len(cov.index.levels[0]))
    cov = pd.DataFrame(val.T, index=cov.index.levels[0], columns=cov.index.levels[1])
    beta = cov / var
    beta = beta.dropna(how='all').fillna(0)

    return beta.drop(columns='SPY')


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

    stck_data = get_time_period(list(dict.fromkeys(strat_param['stock_list'])), freq=strat_param['freq'],
                                time_peri=time_period)

    stck_data = stck_data.pct_change()
    stck_data_corr = stck_data.rolling(roll).corr()

    stck_data_corr = stck_data_corr.dropna()

    re_index = stck_data_corr.index.levels[0][roll:]

    cov_all = np.stack(
        [stck_data_corr.loc[x] for x in re_index])

    eig = np.linalg.eigh(cov_all)

    eig_vec = eig.eigenvectors[:, -n:, :]

    return eig_vec


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
