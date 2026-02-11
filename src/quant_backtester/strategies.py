from market_filters_analysis import *


# Used to generate signals as related to cointegration of a stock(s) pair
def get_signals(strat_param, stck_data) -> pd.DataFrame:
    stock_one, stock_two = strat_param['stock_list']
    args = strat_param['parameters_']
    rolling = args[1]
    z_threshold = args[0]
    init_money = strat_param['init_money']

    r = stck_data[stock_one].rolling(rolling).cov(stck_data[stock_two])
    var = stck_data[stock_two].rolling(rolling).var()
    beta = r / var
    stck_data_diff = stck_data[stock_one] - beta * stck_data[strat_param['stock_list'][1]]

    rolling_obj_diff = stck_data_diff.rolling(rolling)
    z_score = (stck_data_diff - rolling_obj_diff.mean()) / rolling_obj_diff.std()

    z_score = z_score.dropna()

    exits = ((z_score > z_threshold) & (z_score.shift(1) < z_threshold)) + 0
    entries = ((z_score < -1 * z_threshold) & (z_score.shift(1) > -1 * z_threshold)) + 0
    init_money = init_money
    a_1 = (init_money / stck_data[strat_param['stock_list'][0]])
    a_2 = ((a_1 * (stck_data[strat_param['stock_list'][0]] / stck_data[strat_param['stock_list'][1]]) * (1 / beta)) + 0)

    entries_exits = a_1 * (entries - exits) + 0
    entries_exits_ = -a_2 * (entries - exits) + 0
    entries_exits = pd.concat([entries_exits, entries_exits_], axis=1)

    entries_exits.columns = strat_param['stock_list'][0:2]
    entries_exits = entries_exits.replace(0, np.nan).ffill().fillna(0, )
    #
    return entries_exits


# Generates the signals for a stock as related to mean reversion
def get_signals_mv(strat_param, stck_data):
    args = strat_param['parameters_']
    rolling = args[1]
    z_threshold = args[0]
    init_money = strat_param['init_money']
    rolling_obj_diff = stck_data.rolling(rolling)
    z_score = (stck_data - rolling_obj_diff.mean()) / rolling_obj_diff.std()

    z_score = z_score.dropna()

    exits = ((z_score > z_threshold) & (z_score.shift(1) < z_threshold)) + 0
    entries = ((z_score < -1 * z_threshold) & (z_score.shift(1) > -1 * z_threshold)) + 0
    entries_exits = (entries - exits) + 0
    entries_exits.columns = strat_param['stock_list']

    data_close = stck_data[strat_param['stock_list']].loc[entries_exits.index]

    sold_ideal = (1 / data_close * init_money).astype(int)
    quantities_practical = entries_exits * sold_ideal
    quantities_practical = quantities_practical.replace(0, np.nan).ffill().fillna(0, )
    return quantities_practical
