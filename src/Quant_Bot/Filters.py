import numpy as np
import plotly.io as io

from market_analy import  get_time_period
from statsmodels.tsa.stattools import adfuller, coint

import optuna, statsmodels.api as m

func = ''

def cointegration_filter(cur_stock,show_graphs=False):
    cur_pair=get_time_period(cur_stock['stock_list'],True, freq=cur_stock['freq'], num_data_points=cur_stock['num_p'],shift=int(cur_stock['shift_parameter'])+1)
    cur_stock = cur_stock['stock_list']

    model = m.OLS((cur_pair[cur_stock[0]] ), m.add_constant( (cur_pair[cur_stock[1]]))).fit()
    results = coint(np.log(cur_pair[cur_stock[0]] ),np.log(cur_pair[cur_stock[1]]))[1]
    if show_graphs:
        return model.resid.rolling(28).mean().vbt.plot(title=tuple(cur_stock[0:2]).__str__()). to_html(include_plotlyjs='cdn',include_mathjax=False,auto_play =False,full_html=False)
    arr = np.array([False])
    if results < .05 + .001 +0:
        arr = np.array([True])
    return arr



