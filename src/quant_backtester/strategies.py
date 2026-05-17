"""
    Supported strategies (classes) are mean reversion, momentum trending, cointegration, and cross asset mean reversion.

    All methods are executed individually for the assets passed into them, or for the defined pair in the situation of cointegration.
    Only raw signals (entries, hold, exits) are returned (1,0,-1).
    Weights allocation is done only in the defined method.

    Note z-scores can be input as a tuple with 4 values or as one float if less specificity is acceptable

    Mean reversion is defined by the following parameters:
        upper entry threshold
            Currently an user defined z-score
        upper exit threshold
            Currently 0 by default
        lower entry threshold
            Currently the negative of the user defined z-score
        lower exit threshold
            Currently 0 by default
        roll

     Momentum trending is defined by the following parameters:
        upper entry threshold
            Currently an user defined z-score
        upper exit threshold
            Currently 0.5 + the user defined z-score
        lower entry threshold
            Currently the negative of the user defined z-score
        lower exit threshold
            Currently -0.5 + the negative of the user defined z-score
        roll

        The current feature is to prioritize entered positions over new possible positions.

     Cross asset mean reversion is defined by the following parameters:
        upper entry threshold
            Currently an user defined z-score
        upper exit threshold
            Currently 0 by default
        lower entry threshold
            Currently the negative of the user defined z-score
        lower exit threshold
            Currently 0 by default

    Cointegration is defined by the following parameters:
        upper entry threshold
            Currently an user defined z-score
        upper exit threshold
            Currently the negative of the user defined z-score
        lower entry threshold
            Currently the upper exit threshold
        lower exit threshold
            Currently the upper entry threshold
        Therefore, currently the positions merely get flipped.
"""

import numpy as np
import pandas as pd

from .market_filters_analysis import *
from .market_filters_analysis import _pc_filter_weights, _beta_filter_weights, \
    _rolling_filter
np.set_printoptions(suppress=False)


def _get_signals_momentum_tr(strat_param, stck_data) -> pd.DataFrame:
    """
    Used to generate signals for the implementation of momentum trending of a given list of assets

    Note:
         - entries are followed by exits, meaning signals in between are merely ignored, and treated as noise
         - raw signals are generated, meaning no weights are applied at this step
    Args:
        strat_param: dict
        stck_data: pd.DataFrame

    Returns: pd.DataFrame
        Data frame of signals
    """

    args = strat_param['parameters_']
    rolling = args['roll']
    z_threshold = args['z_threshold']
    if type(z_threshold) == tuple:
        long_exit_threshold = z_threshold[1]
        short_exit_threshold = -z_threshold[-1]
        long_entry_threshold = z_threshold[0]
        short_entry_threshold = -z_threshold[-2]
    else:
        long_exit_threshold = z_threshold + 0.5
        short_exit_threshold = -z_threshold - 0.5
        long_entry_threshold = z_threshold
        short_entry_threshold = -z_threshold
    stck_data = stck_data.pct_change()
    rolling_obj_diff = stck_data.rolling(rolling)
    z_score = (stck_data - rolling_obj_diff.mean()) / rolling_obj_diff.std()

    z_score = z_score.dropna()


    short_entries = ((z_score <  short_entry_threshold) &   (z_score.shift(1) > short_entry_threshold)  ) + 0
    short_exits = ((z_score <  short_exit_threshold) & (    short_entry_threshold> z_score.shift(1)) ) + 0
    entries_ = (short_entries - short_exits).replace(0, np.nan).ffill().fillna(0, ).replace(-1,0)

    entries   = ((z_score >  long_entry_threshold) &    (z_score.shift(1) < long_entry_threshold) ) + 0
    long_exits =((z_score >  long_exit_threshold) &(  z_score.shift(1) < long_exit_threshold) ) + 0
    t_entries = ( entries - long_exits).replace(0, np.nan).ffill().fillna(0, ).replace(-1,0)
    sig = t_entries - entries_
    sig[(entries_ *t_entries) != 0] = np.nan
    sig = sig.ffill().fillna(0)

    return       sig


def _get_signals(strat_param, stck_data_full) -> pd.DataFrame:
    """
    To generate raw signals for a cointegration strategy.


    Note:
        Works for single and multi-pair assets
        Entry format is a list of assets with every even index asset being cointegrated with respect to the next odd index
    asset.
    One asset can be paired with many others, but each relationship will be explicitly inputted, meaning assets
        can be repeated in the above list. If there is a conflict of overdetermination, the current default is to merely
    pick the first occurring pair in the list.

    Args:
        strat_param: dict
        stck_data_full: pd.DataFrame
            Data frame of data over the given time period for the given assets.

    Returns: pd.DataFrame
        Data frame of signals
    """

    stk_list = strat_param['stock_list']

    stk_list_final = []
    stck_l = list(zip(stk_list[::2], stk_list[1::2]))
    while stck_l:
        cur = stck_l.pop()

        if not stk_list_final:
            stk_list_final.append(cur)
            continue

        cur_ = []
        stk_ = stk_list_final.copy()
        for x in range(len(stk_)):

            if cur[0] in stk_[x] or cur[1] in stk_[x] :
                stk_list_final.remove(stk_[x])
                cur_+= stk_[x]
        cur_ += cur
        cur = list(dict.fromkeys(list(cur_) ))
        stk_list_final.append(list( cur))
        if not stck_l: break

    full_entries_ex = pd.DataFrame()

    strat_param['weights_filter']['cointegration'] = dict(groups=stk_list_final)
    for stock_one_  in stk_list_final:
        stock_one = stock_one_[0]
        for stock_two in stock_one_[1:]:

            stck_data = stck_data_full[[stock_one, stock_two]]
            args = strat_param['parameters_']
            rolling = args['roll']
            z_threshold = args['z_threshold']

            if type(z_threshold) == tuple:
                long_exit_threshold = z_threshold[1]
                short_exit_threshold = -z_threshold[-1]
                long_entry_threshold = z_threshold[0]
                short_entry_threshold = -z_threshold[-2]
            else:
                long_exit_threshold = -z_threshold
                short_exit_threshold = z_threshold
                long_entry_threshold = z_threshold
                short_entry_threshold = -z_threshold
            r = stck_data[stock_one].rolling(rolling).cov(stck_data[stock_two])
            var = stck_data[stock_two].rolling(rolling).var()
            beta = r / var
            stck_data_diff = stck_data[stock_one] - beta * stck_data[stock_two]

            rolling_obj_diff = stck_data_diff.rolling(rolling)
            z_score = (stck_data_diff - rolling_obj_diff.mean()) / rolling_obj_diff.std()
            z_score = z_score.dropna()

            short_entries = ((z_score > short_entry_threshold) & (
                (z_score.shift(1) < short_entry_threshold))) + 0
            short_exits = ((z_score < short_exit_threshold) & (
                    z_score.shift(1) > short_exit_threshold)) + 0

            entries_ = (short_entries - short_exits).replace(0, np.nan).ffill().fillna(0, ).replace(-1, 0)
            entries = ((z_score < long_entry_threshold) &
                       (z_score.shift(1) > long_entry_threshold)) + 0
            long_exits = ((z_score > long_exit_threshold) & (
                    z_score.shift(1) < long_exit_threshold)) + 0
            t_entries = (entries - long_exits).replace(0, np.nan).ffill().fillna(0, ).replace(-1, 0)
            sig = t_entries - entries_
            sig[(entries_ * t_entries) != 0] = np.nan
            sig = sig.ffill().fillna(0)
            a_12 = (1 / beta) + 0

            entries_exits_2 = a_12.loc[sig.index] * sig

            if stock_one not in full_entries_ex.columns:

                entries_exits = pd.concat([sig, entries_exits_2], axis=1)

                entries_exits.columns = [stock_one, stock_two]
                entries_exits = entries_exits.replace(0, np.nan).ffill().fillna(0, )

                full_entries_ex = pd.concat([full_entries_ex, entries_exits], axis=1)
            else:
                entries_exits_2.name = stock_two

                entries_exits = entries_exits_2.replace(0, np.nan).ffill().fillna(0, )

                full_entries_ex = pd.concat([full_entries_ex, entries_exits], axis=1)

    return full_entries_ex


def _weights_alloc(strat_param, raw_entries_exits, filters=None) -> pd.DataFrame:
    """
    Defines the weights of the portfolio

    Note:
        Allocates weights to reduce the number of possibilities of the ideal weight vector at each time step
        Automatically normalizes weights by assumption of a fully invested portfolio for back testing purposes
        Supports the following rolling filters and constraints.
            Rolling filters are implemented market_filters_analysis._rolling_filter
            PC(n) neutrality, beta neutrality, dollar neutrality

    Args:
        strat_param: dict
        raw_entries_exits: pd.DataFrame
          Non-constrained, non-normalized signals of the given assets.
        filters: dict
            Filters to apply to the data if any in order to constrain the weights.

    Returns: pd.DataFrame
        Constrained, normalized signals of the given assets.

    """

    if 'regime_estimator' in filters.keys():
        filters.pop('regime_estimator')
    stk_list = raw_entries_exits.columns
    strat_param['stock_list'] = stk_list

    if filters:
        filters_length = 0

        if 'rebalance' in filters.keys():
            rebalance = filters.pop('rebalance')
            if rebalance:
                filters_length=len(rebalance['metrics'])
                if 'sector' in rebalance['metrics']  :
                    sectors = get_info(['sectorKey'])  .loc[stk_list  ].groupby('sectorKey').groups
                    filters_length += len(sectors.keys()) - 1
        else:

            rebalance = False

        contains_cont = list(strat_param['strat_class'].keys())[0] == 'cointegration'
        if contains_cont:
            f_cointegration = filters.pop('cointegration')
            groups_cointegration = f_cointegration['groups']
            filters_length += sum([len(x)-1 for x in groups_cointegration])

        filters_length += len(filters)
        if filters_length < len(raw_entries_exits.columns):

            if filters_length > 0:
                index = 0
                arr_weights_constraint = np.zeros(
                    (len(raw_entries_exits.index), filters_length, len(raw_entries_exits.columns)))

                if contains_cont:
                    cointegration_data = f_cointegration['data']
                    shapes = cointegration_data.shape[0], arr_weights_constraint.shape[0]
                    diff = np.abs(shapes[1] - shapes[0])
                    max_arg = np.argmax(shapes)
                    if max_arg == 0:
                        cointegration_data = cointegration_data[diff:]
                    else:
                        arr_weights_constraint = arr_weights_constraint[diff:]

                    for x in groups_cointegration:
                        idx = cointegration_data.columns.get_loc(x[0])
                        for y in x[1:]:
                            id_y = cointegration_data.columns.get_loc(y)

                            arr_weights_constraint[:, index, idx] = cointegration_data[x[0]]
                            arr_weights_constraint  [:,index , id_y ] = -cointegration_data[y]

                            index += 1

                if rebalance:
                    ideal_assets = _rolling_filter(strat_param,  filters_r=rebalance)
                    shapes = ideal_assets.shape[0], arr_weights_constraint.shape[0]
                    diff = np.abs(shapes[1] - shapes[0])
                    max_arg = np.argmax(shapes)
                    if max_arg == 0:
                        ideal_assets = ideal_assets[diff:]
                    else:
                        arr_weights_constraint = arr_weights_constraint[diff:]
                    re_len = len(rebalance['metrics'])
                    arr_weights_constraint[:, index:index + re_len, :] =  ideal_assets
                    index += re_len

                for f in filters.keys():
                    if f == 'pc':
                        pc_eigenvectors = _pc_filter_weights(strat_param, )
                        shapes = pc_eigenvectors.shape[0], arr_weights_constraint.shape[0]
                        diff = np.abs(shapes[1] - shapes[0])
                        max_arg = np.argmax(shapes)
                        if max_arg == 0:
                            pc_eigenvectors = pc_eigenvectors[diff:]
                        else:
                            arr_weights_constraint = arr_weights_constraint[diff:]
                        n = strat_param['weights_filter']['pc']['n']
                        arr_weights_constraint[:, index:index + n, :] = pc_eigenvectors
                        index += n

                    elif f == 'beta':
                        beta = _beta_filter_weights(strat_param)
                        shapes = beta.shape[0], arr_weights_constraint.shape[0]
                        diff = np.abs(shapes[1] - shapes[0])
                        max_arg = np.argmax(shapes)
                        if max_arg == 0:
                            beta = beta[diff:]
                        else:
                            arr_weights_constraint = arr_weights_constraint[diff:]
                        arr_weights_constraint[:, index, :] = beta

                        index += 1
                    elif f == 'dollar':
                        ones = np.array([[1] * len(raw_entries_exits.columns)] * arr_weights_constraint.shape[0])
                        arr_weights_constraint[:, index, :] = ones

                        index += 1
                ind_size = arr_weights_constraint.shape[0]
                weights_null = pd.DataFrame(index=raw_entries_exits.index[-ind_size:],
                                            data=np.zeros(
                                                (len(raw_entries_exits.index[-ind_size:]),
                                                 len(raw_entries_exits.columns))),
                                            columns=raw_entries_exits.columns)

                for x in range(len(arr_weights_constraint[:, 0, 0])):
                    y_least_squa = np.linalg.svd(arr_weights_constraint[x, :, :])[2]

                    null_space = y_least_squa[:,-(arr_weights_constraint.shape[2] - arr_weights_constraint.shape[ 1 ]):]

                    ideal_weights = raw_entries_exits.iloc[x].values
                    p = np.linalg.matmul(null_space.T, ideal_weights)
                    p_final = np.linalg.matmul(null_space,p)

                    weights_null.iloc[x] = p_final

                entries_exits = weights_null

                entries_exits = entries_exits.div(entries_exits.abs().sum(axis=1 ), axis=0)

        else:
            entries_exits = raw_entries_exits / len(raw_entries_exits.columns)

    else:
        entries_exits = raw_entries_exits / len(raw_entries_exits.columns)

    return entries_exits


def _get_signals_mv(strat_param, stck_data) -> pd.DataFrame:
    """
    Used to generate signals for the implementation of mean r.

    Note:
         - entries are followed by exits, meaning signals in between are merely ignored, and treated as noise
         - raw signals are generated, meaning no weights are applied at this step

    Args:
        strat_param: dict
        stck_data: pd.DataFrame

    Returns: pd.DataFrame
        Data frame of signals
    """
    args = strat_param['parameters_']
    rolling = args['roll']
    z_threshold = args['z_threshold']
    if type(z_threshold) == tuple:
        long_exit_threshold = z_threshold[1]
        short_exit_threshold = -z_threshold[-1]
        long_entry_threshold = z_threshold[0]
        short_entry_threshold = -z_threshold[-2]
    else:
        long_exit_threshold = -z_threshold
        short_exit_threshold = z_threshold
        long_entry_threshold = z_threshold
        short_entry_threshold = -z_threshold

    rolling_obj_diff = stck_data.rolling(rolling)
    z_score = (stck_data - rolling_obj_diff.mean()) / rolling_obj_diff.std()

    z_score = z_score.dropna()
    # long_exit_threshold = -z_threshold
    # short_exit_threshold = z_threshold
    # long_entry_threshold = z_threshold
    # short_entry_threshold = -z_threshold

    short_entries = ((z_score > short_entry_threshold) & (
                 (z_score.shift(1) < short_entry_threshold))) + 0
    short_exits = ((z_score < short_exit_threshold) & (
            z_score.shift(1) > short_exit_threshold)  )    + 0

    entries_ = (short_entries - short_exits).replace(0, np.nan).ffill().fillna(0, ).replace(-1, 0)
    entries = ((z_score < long_entry_threshold) &
                   (z_score.shift(1) >  long_entry_threshold))+ 0
    long_exits = ((z_score > long_exit_threshold) & (
                 z_score.shift(1) < long_exit_threshold))+ 0
    t_entries = (entries - long_exits).replace(0, np.nan).ffill().fillna(0, ).replace(-1, 0)
    sig = t_entries - entries_
    sig[(entries_ * t_entries) != 0] = np.nan
    sig  = sig.ffill().fillna(0)
    return sig


def _get_signals_mv_cross_asset(strat_param, ) -> pd.DataFrame:
    """
    Used to generate signals for the implementation of cross asset mean reversion.

    Note:
         - entries are followed by exits, meaning signals in between are merely ignored, and treated as noise
         - raw signals are generated, meaning no weights are applied at this step

    Args:
        strat_param: dict

    Returns: pd.DataFrame
        Data frame of signals
    """
    path = Path(__file__).parents[2]
    path = str(path)
    stck_list = pd.read_parquet(path + '/data/processed/close_1d_10y.parquet').columns

    time_period = strat_param['time_period']

    z_threshold = strat_param['parameters_']['z_threshold']
    if type(z_threshold) == tuple:
        long_exit_threshold = z_threshold[1]
        short_exit_threshold = -z_threshold[-1]
        long_entry_threshold = z_threshold[0]
        short_entry_threshold = -z_threshold[-2]
    else:
        long_exit_threshold = z_threshold
        short_exit_threshold = -z_threshold
        long_entry_threshold = -z_threshold
        short_entry_threshold = z_threshold
    stck_data_p = get_time_period(stck_list, freq=strat_param['freq'], time_peri=time_period)
    stck_data = stck_data_p.pct_change()
    stck_data_cr = stck_data.T

    std = stck_data_cr.std()
    mean = stck_data_cr.mean()
    z_score = (stck_data_cr - mean) / std
    z_score = z_score.T
    z_score = z_score[list(dict.fromkeys  ( strat_param['stock_list']))].dropna()


    short_entries = ((z_score > short_entry_threshold) & (
        (z_score.shift(1) < short_entry_threshold))) + 0
    short_exits = ((z_score < short_exit_threshold) & (
            z_score.shift(1) > short_exit_threshold)) + 0

    entries_ = (short_entries - short_exits).replace(0, np.nan).ffill().fillna(0, ).replace(-1, 0)
    entries = ((z_score < long_entry_threshold) &
               (z_score.shift(1) > long_entry_threshold)) + 0
    long_exits = ((z_score > long_exit_threshold) & (
            z_score.shift(1) < long_exit_threshold)) + 0
    t_entries = (entries - long_exits).replace(0, np.nan).ffill().fillna(0, ).replace(-1, 0)
    sig = t_entries - entries_
    sig[(entries_ * t_entries) != 0] = np.nan
    sig  = sig.ffill().fillna(0)

    return sig


def _get_signals_momentum_cross_asset(strat_param) -> pd.DataFrame:
    """
    Used to generate signals for the implementation of cross asset momentum.

    Note:
         - entries are followed by exits, meaning signals in between are merely ignored, and treated as noise
         - raw signals are generated, meaning no weights are applied at this step

    Args:
        strat_param: dict

    Returns: pd.DataFrame
        Data frame of signals
    """
    path = Path(__file__).parents[2]
    path = str(path)
    stck_list = pd.read_parquet(path + '/data/processed/close_1d_10y.parquet').columns

    time_period = strat_param['time_period']

    z_threshold = strat_param['parameters_']['z_threshold']
    if type(z_threshold) == tuple:
        long_exit_threshold = z_threshold[1]
        short_exit_threshold = -z_threshold[-1]
        long_entry_threshold = z_threshold[0]
        short_entry_threshold = -z_threshold[-2]
    else:
        long_exit_threshold = z_threshold + 0.5
        short_exit_threshold = -z_threshold - 0.5
        long_entry_threshold = z_threshold
        short_entry_threshold = -z_threshold
    stck_data_p = get_time_period(stck_list, freq=strat_param['freq'], time_peri=time_period)
    stck_data = stck_data_p.pct_change()
    stck_data_cr = stck_data.T

    std = stck_data_cr.std()
    mean = stck_data_cr.mean()
    z_score = (stck_data_cr - mean) / std
    z_score = z_score.T
    z_score = z_score[list(dict.fromkeys(   strat_param['stock_list']))].dropna()


    short_entries = ((z_score < short_entry_threshold) & (z_score.shift(1) > short_entry_threshold)) + 0
    short_exits = ((z_score < short_exit_threshold) & (short_entry_threshold > z_score.shift(1))) +  0

    entries = ((z_score > long_entry_threshold) & (z_score.shift(1) < long_entry_threshold))  + 0
    long_exits = ((z_score > long_exit_threshold) & (z_score.shift(1) < long_exit_threshold))   + 0
    entries_ = (short_entries - short_exits).replace(0, np.nan).ffill().fillna(0, ).replace(-1, 0)
    t_entries = (entries - long_exits).replace(0, np.nan).ffill().fillna(0, ).replace(-1, 0)
    sig = t_entries - entries_
    sig[(entries_ * t_entries) != 0] = np.nan
    sig = sig.ffill().fillna(0)

    return sig
