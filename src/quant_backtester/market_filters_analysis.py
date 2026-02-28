from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as m
import statsmodels.tsa.stattools
from statsmodels.tsa.stattools import coint

from quant_backtester.data_filter import get_time_period
"""
In general, there are 4 classes of filters. 
    State-determining filters to classify regimes 
        _mean_filter_regime
        _momentum_filter_regime
    Pre-training filters to determine the assets universe.
        _corr_filter
        _market_cap_filter
        _volatility_filter
        _cointegration_filter
        _mv_filter_cross_assets
    Rolling filters to determine which assets to assign non zero weights executed every timestep
        _rolling_filter (highest z_score, lowest/highest correlation, lowest standard deviation, highest mean, most positive mean count)
        Note all are on returns of assets 
    Weight filters to help assign weights depending on the defined aspects of a portfolio.
        _pc_filter_weights (first n eigenvectors)
        _beta_filter_weights 
        dollar neutrality is implemented in _weights_alloc
"""

def _rolling_filter(strat_param, n, filters_r=None):
    """
    Applies rolling filters as defined above.
        :param strat_param: dict
            Strategy parameters, all of which are mayhaps not necessary to use for the given list of filters.
        :param n: int
            Determines the number of top-ranked assets to keep.
        :param filters_r: dict
            List of filters as keys with any associated parameters as the values.
        :return: pd.DataFrame
            Returns a boolean dataframe with the same axes as the signals dataframe. Indicates whether to include the asset,
            which are listed as the columns.
    """


    time_period = strat_param['time_period']
    z_threshold = strat_param['parameters_']['z_threshold']
    roll = strat_param['parameters_']['roll']
    stck_data = get_time_period(list(set(strat_param['stock_list'])), True, freq=strat_param['freq'], time_peri=time_period)

    sort_max = True
    stck_data = stck_data.pct_change()
    for type_ in filters_r:
        if type_ == 'z_score':

            std = stck_data.rolling(roll).std()
            mean = stck_data.rolling(roll).mean()
            z_score = (stck_data - mean) / std
            filter_ser = z_score.rolling(roll).std().dropna()
        elif type_ == 'std':
            filter_ser = stck_data.rolling(roll).std().dropna()
            sort_max = False
        elif type_ == 'corr':
            filter_ser = stck_data.rolling(roll).corr().dropna().mean(axis=1)
        elif type_ == 'mean':
            filter_ser = stck_data.rolling(roll).mean().dropna()
        elif type_ == 'mean_inc_v_d':
            mean=stck_data.rolling(roll).mean()
            filter_ser = mean.where(mean > 0,axis=1).count(axis = 1)

        else:
            filter_ser = stck_data.rolling(roll).mean().dropna() / stck_data.rolling(roll).std().dropna()

    z_rk = np.array([[False] * len(filter_ser.columns)] * len(filter_ser))
    top_k = np.array([len(filter_ser.columns), n]).min()
    if not sort_max:
        indx = np.argpartition(filter_ser.values, top_k,axis=1)

        np.put_along_axis(z_rk,
                      indx[:,:top_k  ], True, 1)
    else:
        indx = np.argpartition(filter_ser.values, -top_k, axis=1)

        np.put_along_axis(z_rk,
                          indx[:, -top_k:], True, 1)

    z_rk = pd.DataFrame(z_rk, index=filter_ser.index, columns=filter_ser.columns)

    return z_rk

def _momentum_filter_regime(strat_param):
    """
    Regime estimator. Trend filter (simple)
        :param strat_param: dict
        :return: Boolean
            Whether the baseline has compounded in value
    """
    base_data = get_time_period(['SPY'], True, freq=strat_param['freq'], time_peri=strat_param['time_period'])
    base_data = base_data.pct_change()

    return (1+base_data).cumprod().iloc[-1].iloc[0] > 1

def regime_estimator(strat_param):
    """
    Helper to determine what strategy to use depending on the regime
         :param strat_param: dict
         :return: String
     """
    if _mean_rev_regime(strat_param) and not _momentum_filter_regime(strat_param):
        return 'mv'
    if _momentum_filter_regime(strat_param):
        return 'momentum_trending'
    else:
        return 'undetermined'
def _mean_rev_regime  (strat_param):
    """
    Regime estimator. Mean rev filter (simple)
        :param strat_param: dict
        :return: Boolean
            Whether the baseline has returns which satisfy the ADF test with a p-value of less than .05
    """
    base_data = get_time_period(['SPY'], True, freq=strat_param['freq'], time_peri=strat_param['time_period'])
    base_data = base_data.pct_change()
    indicator = statsmodels.tsa.stattools.adfuller(base_data.dropna())[1]
    return indicator < .05
def market_cap_filter(strat_param, type_):
    """
    Pre-training filter. Market cap filter to classify assets based on average market cap over the given time period
        :param type_: str
            Whether to return 'small', 'mid', 'large' cap assets
        :param strat_param: dict
        :return: np.array
            Array of stocks which pass the given filter
    """
    time_period = strat_param['time_period']
    base_data = get_time_period(strat_param['stock_list'], True, freq=strat_param['freq'], time_peri=time_period)
    market_cap = base_data * get_time_period(strat_param['stock_list'], True, freq=strat_param['freq'],
                                             time_peri=time_period, type_='Volume')
    market_cap = market_cap.mean()
    m_large_threshold = 10 * 10 ** 9
    m_small_threshold = 2 * 10 ** 9
    m_medium_threshold = 300 * 10 ** 6

    if type_ == 'large':
        return market_cap.where(market_cap > m_large_threshold).dropna().index
    elif type_ == 'medium':
        return market_cap.where((m_medium_threshold < market_cap) & (market_cap < m_large_threshold)).dropna().index
    elif type_ == 'small':
        return market_cap.where((m_small_threshold < market_cap) + (m_medium_threshold > market_cap)).dropna().index
    return None


def volatility_filter(strat_param, q_threshold):
    """
    Pre-training filter. Volatility filter which returns all assets which fall below a given quantile
        :param q_threshold: float
            What quantile of assets to return as classified by average variance
        :param strat_param: dict
        :return: np.array
            Array of stocks which pass the given filter
    """
    time_period = strat_param['time_period']
    base_data = get_time_period(strat_param['stock_list'], True, freq=strat_param['freq'], time_peri=time_period)
    var = base_data.var()

    var = var.where(var < var.quantile(q_threshold)).dropna()

    return np.array(var.index)



def cointegration_filter(strat_param, show_graphs=False):
    """
    Executes the Engel-Granger test on a pair of assets to determine if cointegrated on the given time period.
        :param show_graphs: Boolean, default is False
        :param strat_param: dict
        :return: np.array
            Array of stocks which pass the given filter
    """
    time_period = strat_param['time_period']
    cur_pair = get_time_period(strat_param['stock_list'], True, freq=strat_param['freq'], time_peri=time_period)
    cur_stock = strat_param['stock_list']

    model = m.OLS((cur_pair[cur_stock[0]]), m.add_constant((cur_pair[cur_stock[1]]))).fit()
    results = coint(np.log(cur_pair[cur_stock[0]]), np.log(cur_pair[cur_stock[1]]))[1]

    if show_graphs:
        return model.resid.rolling(28).mean().vbt.plot(title=tuple(cur_stock[0:2]).__str__()).to_html(
            include_plotlyjs='cdn', include_mathjax=False, auto_play=False, full_html=False)
    arr = np.array([False])

    if results < .05 + .001:
        arr = np.array([True])
    return arr


def _beta_filter_weights(strat_param):
    """
    Weights filter. Returns a rolling collection of beta values for each given asset, as related to the baseline, over
    the given time period.
        :param strat_param: dict
        :return: pd.DataFrame
            Data frame of beta values for each asset indexed by timestep
    """
    time_period = strat_param['time_period']
    stck_data = get_time_period(list(set(strat_param['stock_list'
                                         ] )) + ['SPY'], True, freq=strat_param['freq'],
                                time_peri=time_period)
    stck_data = stck_data.pct_change().dropna()

    roll = strat_param['parameters_']['roll']
    cov = stck_data.rolling(roll).cov()['SPY']
    var = stck_data.rolling(roll).var()

    val = cov.values.reshape(len(cov.index.levels[1]), len(cov.index.levels[0]))
    cov = pd.DataFrame(val.T, index=cov.index.levels[0], columns=cov.index.levels[1])
    beta = cov / var
    beta = beta.dropna(how='all').fillna(0)

    return beta.drop(columns='SPY')

def _pc_filter_weights(strat_param, n=1):
    """
    Weights filter. Returns the n top eigenvectors of the correlation matrix over the given time period, with the given
    roll.
        :param strat_param: dict
        :param n: int
        :return: np.array
    """
    time_period = strat_param['time_period']
    roll = strat_param['parameters_']['roll']

    stck_data = get_time_period(strat_param['stock_list'], True, freq=strat_param['freq'], time_peri=time_period)
    stck_data = stck_data.pct_change()
    stck_data_corr = stck_data.rolling(roll).corr()
    stck_data_corr = stck_data_corr.dropna()

    cov_all = np.stack(
        [stck_data_corr.loc[x] for x in stck_data_corr.index.levels[0] if x in [y[0] for y in stck_data_corr.index]])

    eig = np.linalg.eigh(cov_all)
    eig_vec = eig.eigenvectors[:, 0:n, :]
    return eig_vec
#
def corr_filter(strat_param, c_threshold):
    """
    Pre-training filter. Returns all assets which have a pairwise correlation less than the given threshold. For those that
    do have a higher correlation, only one of the assets is returned.
        :param c_threshold: float
        :param strat_param: dict
        :return: np.array
            Array of stocks which pass the given filter
    """
    time_period = strat_param['time_period']
    base_data = get_time_period(strat_param['stock_list'], True, freq=strat_param['freq'], time_peri=time_period)
    corr = base_data.corr()
    corr_lower = np.tril(corr, k=-1)
    corr = pd.DataFrame(corr_lower, index=corr.index, columns=corr.columns)

    corr = corr.where(corr.abs() < +  c_threshold).dropna()

    return np.array(corr.index)


def mv_filter_cross_assets(strat_param):
    """

    :param strat_param: dict
    :return:
    """
    time_period = strat_param['time_period']
    roll = strat_param['parameters_']['roll']

    stck_data = get_time_period(strat_param['stock_list'], True, freq=strat_param['freq'], time_peri=time_period)
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
        :param results:
        :param kwargs:
        :return: None
    """
    track = []
    html_ = [
        '<html style="display: flex; justify-content: center"><body style="background-color:#1a1a1a ;  color:white">']
    results = results.index
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

    path_name = str(path) + '/docs/results/' + kwargs['strat_class'] + '/' + str(kwargs['time_period']) + '.html'
    Path(path_name).write_text('\n'.join(html_), encoding='utf-8')
