import numpy as np

from Market_Analysis import  get_time_period
from statsmodels.tsa.stattools import adfuller, coint

import statsmodels.api as m

# To determine cointegration of a single stock pair given a time period.
def cointegration_filter(strat_param,show_graphs=False):

    cur_pair=get_time_period(strat_param['stock_list'],True, freq=strat_param['freq'], num_data_points=strat_param['num_p'],shift=int(strat_param['shift_parameter'])+1)
    cur_stock = strat_param['stock_list']

    model = m.OLS((cur_pair[cur_stock[0]] ), m.add_constant( (cur_pair[cur_stock[1]]))).fit()
    results = coint(np.log(cur_pair[cur_stock[0]] ),np.log(cur_pair[cur_stock[1]]))[1]

    if show_graphs:
        return model.resid.rolling(28).mean().vbt.plot(title=tuple(cur_stock[0:2]).__str__()). to_html(include_plotlyjs='cdn',include_mathjax=False,auto_play =False,full_html=False)
    arr = np.array([False])

    if results < .05 + .001 +0:
        arr = np.array([True])
    return arr



