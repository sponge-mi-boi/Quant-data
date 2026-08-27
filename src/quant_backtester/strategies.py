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

_CROSS_ASSET_SIGNAL_CACHE = {}


def _project_onto_constraint_nullspace(signal, constraints, tolerance=None):
    """Return the least-squares projection of a signal onto Cw=0."""
    signal = np.asarray(signal, dtype=float)
    constraints = np.asarray(constraints, dtype=float)
    if constraints.ndim != 2 or constraints.shape[1] != signal.shape[0]:
        raise ValueError('constraints must have shape (constraint_count, asset_count)')
    if not np.isfinite(constraints).all() or not np.isfinite(signal).all():
        return np.zeros_like(signal)
    _, singular_values, vh = np.linalg.svd(constraints, full_matrices=True)
    if tolerance is None:
        largest = singular_values[0] if len(singular_values) else 0.0
        tolerance = max(constraints.shape) * np.finfo(float).eps * largest
    rank = int(np.sum(singular_values > tolerance))
    basis = vh[rank:].T
    if basis.shape[1] == 0:
        return np.zeros_like(signal)
    return basis @ (basis.T @ signal)


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
        # Enter on an extreme return and exit after momentum weakens.
        # Keeping the exit threshold inside the entry threshold produces
        # conventional hysteresis and prevents exits from requiring an even
        # more extreme move in the same direction.
        long_exit_threshold = max(0.0, z_threshold - 0.5)
        short_exit_threshold = min(0.0, -z_threshold + 0.5)
        long_entry_threshold = z_threshold
        short_entry_threshold = -z_threshold
    stck_data = stck_data.pct_change()
    rolling_obj_diff = stck_data.rolling(rolling)
    z_score = (stck_data - rolling_obj_diff.mean()) / rolling_obj_diff.std()

    z_score = z_score.dropna()


    short_entries = ((z_score <  short_entry_threshold) &   (z_score.shift(1) > short_entry_threshold)  ) + 0
    short_exits = ((z_score > short_exit_threshold) &
                   (z_score.shift(1) <= short_exit_threshold)) + 0
    entries_ = (short_entries - short_exits).replace(0, np.nan).ffill().fillna(0, ).replace(-1,0)

    entries   = ((z_score >  long_entry_threshold) &    (z_score.shift(1) < long_entry_threshold) ) + 0
    long_exits = ((z_score < long_exit_threshold) &
                  (z_score.shift(1) >= long_exit_threshold)) + 0
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

    stk_list = list(strat_param['stock_list'])
    if len(stk_list) == 0 or len(stk_list) % 2:
        raise ValueError('cointegration stock_list must contain complete asset pairs')
    if len(set(stk_list)) != len(stk_list):
        raise ValueError('cointegration pairs may not share assets in one portfolio')

    args = strat_param['parameters_']
    rolling = int(args['roll'])
    if rolling < 3:
        raise ValueError('cointegration roll must be at least 3')
    z_threshold = args['z_threshold']
    pair_weights = []

    for stock_one, stock_two in zip(stk_list[::2], stk_list[1::2]):
        prices = stck_data_full[[stock_one, stock_two]].dropna()
        if (prices <= 0).any().any():
            raise ValueError('cointegration requires strictly positive prices for log transformation')
        log_prices = np.log(prices)
        y = log_prices[stock_one]
        x = log_prices[stock_two]

        beta = y.rolling(rolling).cov(x).div(x.rolling(rolling).var()).replace([np.inf, -np.inf], np.nan)
        alpha = y.rolling(rolling).mean() - beta * x.rolling(rolling).mean()
        spread = y - alpha - beta * x
        spread_window = spread.rolling(rolling)
        z_score = ((spread - spread_window.mean()) / spread_window.std()).replace([np.inf, -np.inf], np.nan).dropna()

        spread_signal = _mean_reversion_signals_from_z(z_score.to_frame('spread'), z_threshold)['spread']
        pair = pd.concat(
            [spread_signal.rename(stock_one), (-beta.loc[spread_signal.index] * spread_signal).rename(stock_two)],
            axis=1,
        ).fillna(0.0)
        pair_weights.append(pair)

    return pd.concat(pair_weights, axis=1).sort_index().fillna(0.0)


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

    filters = {} if filters is None else filters.copy()
    if 'regime_estimator' in filters.keys():
        filters.pop('regime_estimator')
    stk_list = raw_entries_exits.columns
    strat_param['stock_list'] = stk_list

    entries_exits = raw_entries_exits.copy()

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

        contains_cont = (list(strat_param['strat_class'].keys())[0] == 'cointegration'
                         and 'cointegration' in filters)
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
                    ideal_weights = raw_entries_exits.iloc[-ind_size + x].values
                    weights_null.iloc[x] = _project_onto_constraint_nullspace(
                        ideal_weights, arr_weights_constraint[x])

                entries_exits = weights_null

                entries_exits = entries_exits.div(entries_exits.abs().sum(axis=1 ), axis=0)

    # Normalize active positions to one unit of gross exposure.  Dividing by
    # the total number of portfolio columns leaves most capital idle whenever
    # only a subset of assets has a signal.
    gross_exposure = entries_exits.abs().sum(axis=1).replace(0, np.nan)
    return entries_exits.div(gross_exposure, axis=0).fillna(0.0)


def _mean_reversion_thresholds(z_threshold):
    """Return positive entry and exit magnitudes for mean reversion."""
    if np.isscalar(z_threshold):
        entry, exit_ = float(z_threshold), 0.0
    else:
        if len(z_threshold) != 2:
            raise ValueError('mean-reversion z_threshold must be a scalar or (entry, exit) pair')
        entry, exit_ = (float(z_threshold[0]), float(z_threshold[1]))
    if not np.isfinite(entry) or not np.isfinite(exit_):
        raise ValueError('mean-reversion thresholds must be finite')
    if entry <= 0 or exit_ < 0 or exit_ >= entry:
        raise ValueError('mean-reversion thresholds must satisfy 0 <= exit < entry')
    return entry, exit_


def _mean_reversion_signals_from_z(z_score, z_threshold) -> pd.DataFrame:
    """Create long/flat/short states from a z-score without overlapping states."""
    entry, exit_ = _mean_reversion_thresholds(z_threshold)
    previous = z_score.shift(1)

    long_entries = (z_score < -entry) & (previous >= -entry)
    short_entries = (z_score > entry) & (previous <= entry)
    long_exits = (z_score >= -exit_) & (previous < -exit_)
    short_exits = (z_score <= exit_) & (previous > exit_)

    signals = pd.DataFrame(np.nan, index=z_score.index, columns=z_score.columns)
    signals = signals.mask(long_exits | short_exits, 0.0)
    # Entries are applied after exits so a gap across both bands can reverse
    # directly into the newly indicated position.
    signals = signals.mask(long_entries, 1.0)
    signals = signals.mask(short_entries, -1.0)
    return signals.ffill().fillna(0.0)


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
    rolling_obj_diff = stck_data.rolling(rolling)
    z_score = (stck_data - rolling_obj_diff.mean()) / rolling_obj_diff.std()
    z_score = z_score.dropna()
    return _mean_reversion_signals_from_z(z_score, z_threshold)


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
    time_period = strat_param['time_period']

    z_threshold = strat_param['parameters_']['z_threshold']
    requested_assets = list(dict.fromkeys(strat_param['stock_list']))
    cache_key = ('cross_asset_mv', tuple(time_period), str(strat_param['freq']),
                 str(z_threshold), tuple(requested_assets))
    stck_list = requested_assets
    if cache_key in _CROSS_ASSET_SIGNAL_CACHE:
        return _CROSS_ASSET_SIGNAL_CACHE[cache_key][requested_assets].copy()
    stck_data_p = get_time_period(stck_list, freq=strat_param['freq'], time_peri=time_period)
    stck_data = stck_data_p.pct_change()
    stck_data_cr = stck_data.T

    std = stck_data_cr.std()
    mean = stck_data_cr.mean()
    z_score = (stck_data_cr - mean) / std
    z_score = z_score.T
    z_score = z_score.dropna()
    signals = _mean_reversion_signals_from_z(z_score, z_threshold)
    _CROSS_ASSET_SIGNAL_CACHE[cache_key] = signals
    return signals[requested_assets].copy()


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
    time_period = strat_param['time_period']

    z_threshold = strat_param['parameters_']['z_threshold']
    requested_assets = list(dict.fromkeys(strat_param['stock_list']))
    cache_key = ('cross_asset_momentum_trending', tuple(time_period), str(strat_param['freq']),
                 str(z_threshold), tuple(requested_assets))
    stck_list = requested_assets
    if cache_key in _CROSS_ASSET_SIGNAL_CACHE:
        return _CROSS_ASSET_SIGNAL_CACHE[cache_key][requested_assets].copy()
    if np.isscalar(z_threshold):
        entry_threshold = float(z_threshold)
        exit_threshold = max(0.0, entry_threshold - 0.5)
    else:
        if len(z_threshold) != 2:
            raise ValueError('cross-sectional momentum z_threshold must be a scalar or (entry, exit) pair')
        entry_threshold, exit_threshold = map(float, z_threshold)
    if not (np.isfinite(entry_threshold) and np.isfinite(exit_threshold)):
        raise ValueError('cross-sectional momentum thresholds must be finite')
    if entry_threshold <= 0 or exit_threshold < 0 or exit_threshold >= entry_threshold:
        raise ValueError('cross-sectional momentum thresholds must satisfy 0 <= exit < entry')
    stck_data_p = get_time_period(stck_list, freq=strat_param['freq'], time_peri=time_period)
    stck_data = stck_data_p.pct_change()
    stck_data_cr = stck_data.T

    std = stck_data_cr.std()
    mean = stck_data_cr.mean()
    z_score = (stck_data_cr - mean) / std
    z_score = z_score.T
    z_score = z_score.dropna()


    previous = z_score.shift(1)
    long_entries = (z_score > entry_threshold) & (previous <= entry_threshold)
    short_entries = (z_score < -entry_threshold) & (previous >= -entry_threshold)
    long_exits = (z_score <= exit_threshold) & (previous > exit_threshold)
    short_exits = (z_score >= -exit_threshold) & (previous < -exit_threshold)

    signals = pd.DataFrame(np.nan, index=z_score.index, columns=z_score.columns)
    signals = signals.mask(long_exits | short_exits, 0.0)
    signals = signals.mask(long_entries, 1.0)
    signals = signals.mask(short_entries, -1.0)
    signals = signals.ffill().fillna(0.0)
    _CROSS_ASSET_SIGNAL_CACHE[cache_key] = signals
    return signals[requested_assets].copy()
