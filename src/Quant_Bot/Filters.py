import numpy as np
from pathlib import Path

from Market_Analysis import  get_time_period
from statsmodels.tsa.stattools import adfuller, coint

import statsmodels.api as m
import pandas as pd

# To determine cointegration of a single stock pair given a time period.
def cointegration_filter(strat_param,show_graphs=False):

    time_period = strat_param['shift_parameter']
    cur_pair=get_time_period(strat_param['stock_list'],True, freq=strat_param['freq'], time_peri = time_period)
    cur_stock = strat_param['stock_list']

    model = m.OLS((cur_pair[cur_stock[0]] ), m.add_constant( (cur_pair[cur_stock[1]]))).fit()
    results = coint(np.log(cur_pair[cur_stock[0]] ),np.log(cur_pair[cur_stock[1]]))[1]

    if show_graphs:
        return model.resid.rolling(28).mean().vbt.plot(title=tuple(cur_stock[0:2]).__str__()). to_html(include_plotlyjs='cdn',include_mathjax=False,auto_play =False,full_html=False)
    arr = np.array([False])

    if results < .05 + .001:
        arr = np.array([True])
    return arr
def get_analysis(results = None, **kwargs):

    track = []
    html_ = ['<html style="display: flex; justify-content: center"><body style="background-color:#1a1a1a ;  color:white">']
    results = results .index
    results = [list(x) for x in results]

    for x in results :
        for y in kwargs[ 'parameters']:
            kwargs['shift_parameter'] = y
            kwargs['stock_list'] = x
            graph = kwargs['filter_func'](
    kwargs,  )
        if graph:
            html_.append(               '<div style="display: flex">'), html_.append(graph[0]), html_.append(graph[1])
            html_.append('</div>'      ),
            html_.append(graph[2]     ),
            html_.append(graph[3]),
            html_.append('<br><br<br><br>')
            track.append(dict(stock=x,elem=html_[-7:],metric = graph[-1]))
    track.sort(key=lambda x: x['metric'])


    html_ = [y  for x in track for y in x['elem'] ]
    html_ = ['<html style="display: flex; justify-content: center"><body style="background-color:#1a1a1a ;  color:white">'] + html_
    html_.append('</body></html>'           ),

    path_name = kwargs['strat_class'] + '/' + str (  kwargs['shift_parameter']) + '.html'
    Path(path_name).write_text('\n'.join(html_), encoding='utf-8')

def mean_rev_filter(stck_list  , time_period, z_score = 1.5 ):
    ex = get_time_period(stck_list,time_peri=time_period)
    ex = ex. pct_change()
    r = pd.Series(index=ex.columns)
    for x in ex.columns:
        exp = ex[x]
        std = exp .rolling(50). std()
        m =  exp .rolling(50). mean ()
        z_score =  ( (exp)      -m)/std
        z_score = z_score[abs(z_score) > z_score ]
        r[x] = len(z_score)
    return r .sort_values() .iloc[-10:]
