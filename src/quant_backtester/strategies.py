import numpy as np
import pandas as pd

from .market_filters_analysis import _pc_filter_weights, _beta_filter_weights, \
    _rolling_filter
from .market_filters_analysis import *



def _get_signals(strat_param, stck_data_full) -> pd.DataFrame:
    """
    Used to generate signals as related to cointegration of a stock(s) pair
        :param strat_param: dict
        :param stck_data_full: pd.DataFrame
            Data frame of data over the given time period for the given assets.
        :return: pd.DataFrame
            Data frame of signals
    """

    stk_list = strat_param['stock_list']

    for stk in range(len(stk_list) ):
         if stk_list. count(stk_list[stk]) > 1:
             if stk % 2 == 1:
                temp = stk_list[stk]
                stk_list[stk] = stk_list[stk - 1]
                stk_list[stk - 1] = temp

    full_entries_ex = pd.DataFrame()
    for pairs in range (0,int(len(strat_param['stock_list'])), 2  ):

        stock_one, stock_two = strat_param['stock_list'][pairs: pairs + 2  ]

        stck_data = stck_data_full[[stock_one,stock_two]]
        args = strat_param['parameters_']
        rolling = args['roll']
        z_threshold = args['z_threshold']


        r = stck_data[stock_one].rolling(rolling).cov(stck_data[stock_two])
        var = stck_data[stock_two].rolling(rolling).var()
        beta = r / var
        stck_data_diff = stck_data[stock_one] - beta * stck_data[ stock_two        ]
#
        rolling_obj_diff = stck_data_diff.rolling(rolling)
        z_score = (stck_data_diff - rolling_obj_diff.mean()) / rolling_obj_diff.std()

        exits = ((z_score > z_threshold) & (z_score.shift(1) < z_threshold)) + 0
        entries = ((z_score < -1 * z_threshold) & (z_score.shift(1) > -1 * z_threshold)) + 0

        a_12 = (1 / beta) + 0
        entries_exits_1 = (entries - exits) + 0
        entries_exits_2 = -a_12 * entries_exits_1

        entries_exits = pd.concat([entries_exits_1, entries_exits_2], axis=1)

        entries_exits.columns = [stock_one,stock_two]
        entries_exits = entries_exits.replace(0, np.nan).ffill().fillna(0, )

        if stock_one in full_entries_ex.columns:
            entries_exits = entries_exits.drop(columns= [stock_one ]  )
        full_entries_ex = pd.concat([full_entries_ex, entries_exits], axis=1)

    return full_entries_ex



def _get_signals_momentum_tr(strat_param, stck_data):
    """
    Used to generate signals as related to momentum trending of a given list of assets
        :param strat_param: dict
        :param stck_data: pd.DataFrame
            Data frame of data over the given time period for the given assets.
        :return: pd.DataFrame
            Data frame of signals
    """
    args = strat_param['parameters_']
    rolling = args['roll']
    z_threshold = args['z_threshold']

    rolling_obj_diff = stck_data.rolling(rolling)
    z_score = (stck_data - rolling_obj_diff.mean()) / rolling_obj_diff.std()

    z_score = z_score.dropna()

    exits = (z_score * z_score.shift(1)) < 0 + 0
    short_entries = ((z_score < -1 * z_threshold) & (z_score.shift(1) > -1 * z_threshold)).astype(int) + 0
    entries = short_entries * -1 + ((z_score > z_threshold) & (z_score.shift(1) < z_threshold))
    entries = entries.where(exits == False, 1000)
    entries.columns = strat_param['stock_list']

    entries = entries.replace(0, np.nan).ffill().fillna(0, )

    entries = entries.replace(1000, 0)

    return entries


def _get_signals_mv(strat_param, stck_data):
    """
    Used to generate signals as related mean reversion the given assets.
        :param strat_param: dict
        :param stck_data: pd.DataFrame
            Data frame of data over the given time period for the given assets.
        :return: pd.DataFrame
            Data frame of signals
    """
    args = strat_param['parameters_']
    rolling = args['roll']
    z_threshold = args['z_threshold']

    rolling_obj_diff = stck_data.rolling(rolling)
    z_score = (stck_data - rolling_obj_diff.mean()) / rolling_obj_diff.std()

    z_score = z_score.dropna()
    exits = (z_score * z_score.shift(1)) < 0 + 0
    l_entries = ((z_score < -1 * z_threshold) & (z_score.shift(1) > -1 * z_threshold)).astype(int) + 0
    entries = l_entries + -1 * ((z_score > z_threshold) & (z_score.shift(1) < z_threshold))
    entries = entries.where(exits == False, 1000)
    entries = entries.replace(0, np.nan).ffill().fillna(0, )
    entries.columns = strat_param['stock_list']

    entries = entries.replace(1000, 0)

    return entries


def _weights_alloc(strat_param, raw_entries_exits, filters=None):
    """

    :param strat_param: dict
    :param raw_entries_exits: pd.DataFrame
        Non-constrained, non-normalized signals of the given assets.
    :param filters: dict
        Filters to apply to the data if any in order to constrain the weights.
    :return: pd.DataFrame
        Constrained, normalized signals of the given assets.
    """
    if filters:
        filters_length = len(filters) - 1
        if strat_param['strat_class'] == 'cointegration':
            filters_length += int(len(strat_param['stock_list']) / 2)

        if    filters_length< len(raw_entries_exits.columns) :
            weights_null = pd.DataFrame(index=raw_entries_exits.index,
                                        data=np.zeros((len(raw_entries_exits.index), len(raw_entries_exits.columns))),
                                        columns=raw_entries_exits.columns)
            rebalance = filters.pop('rebalance')

            if filters_length > 0:
                index = 0
                arr_weights_constraint = np.zeros (
                    (len(raw_entries_exits.index), filters_length, len(raw_entries_exits.columns)))

                if strat_param['strat_class'] == 'cointegration':
                    arr_weights_constraint[:, index, :] = raw_entries_exits. values
                    index += 1

                if rebalance:
                    ideal_assets = _rolling_filter(strat_param, 5, filters_r=filters) + 0

                    ideal_assets, raw_entries_exits = ideal_assets.align(raw_entries_exits,join='right')
                    # ideal_assets = raw_entries_exits + raw_entries_exits * ideal_assets

                    arr_weights_constraint[:, index, :] = ideal_assets
                    index += 1

                for f in  filters.keys() :
                    if f == 'pc' :
                        pc_eigenvectors = _pc_filter_weights(strat_param)
                        pc_eigenvectors = np.vstack([pc_eigenvectors,np.zeros((1,1,     len(raw_entries_exits.columns)))])
                        arr_weights_constraint[:, index, :] = pc_eigenvectors[:, 0, :]

                    elif f == 'beta':
                        beta = _beta_filter_weights(strat_param)

                        beta, raw_entries_exits = beta.align(raw_entries_exits,join='right')
                        beta, = beta.fillna(0),

                        arr_weights_constraint[:, index, :] = beta

                    elif f == 'dollar':
                        ones = np.array([[1] * len(raw_entries_exits.columns)] * len(beta.index))
                        arr_weights_constraint[:, index, :] = ones
                for x in range(len(arr_weights_constraint[:, 0, 0  ])):


                    y_least_squa = np.linalg.svd(arr_weights_constraint[x, :, :])[2]

                    weights_null.iloc[x] = tuple(y_least_squa[:,   0])
                entries_exits = weights_null
        else:
            entries_exits = raw_entries_exits

    else:
        entries_exits = raw_entries_exits
    entries_exits = entries_exits.div(entries_exits.abs().sum(axis=1), axis=0).fillna(0)

    return entries_exits



def _get_signals_mv_cross_asset(strat_param, ):
    """
     Used to generate signals as related to cross-sectional mean reversion of a given list of assets
         :param strat_param: dict
             Data frame of data over the given time period for the given assets.
         :return: pd.DataFrame
             Data frame of signals
     """
    path = Path(__file__).parents[2]
    path = str(path)
    stck_list = pd.read_parquet(path + '/data/processed/close_1d_10y.parquet').columns

    time_period = strat_param['time_period']

    z_threshold = strat_param['parameters_']['z_threshold']

    stck_data_p = get_time_period(stck_list, True, freq=strat_param['freq'], time_peri=time_period)
    stck_data = stck_data_p.pct_change()
    stck_data_cr = stck_data.T

    std = stck_data_cr.std()
    mean = stck_data_cr.mean()
    z_score = (stck_data_cr - mean) / std
    z_score = z_score.T
    z_score = z_score[strat_param['stock_list']].dropna()

    z_score = z_score.dropna()
    exits = (z_score * z_score.shift(1)) < 0 + 0
    l_entries = ((z_score < -1 * z_threshold) & (z_score.shift(1) > -1 * z_threshold)).astype(int) + 0
    entries = l_entries + -1 * ((z_score > z_threshold) & (z_score.shift(1) < z_threshold))
    entries.columns = strat_param['stock_list']
    entries = entries.where(exits == False, 1000)
    entries = entries.replace(0, np.nan).ffill().fillna(0, )
    entries = entries.replace(1000, 0)
    return entries
