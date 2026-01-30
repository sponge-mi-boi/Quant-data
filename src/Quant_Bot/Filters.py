import numpy as np
from pathlib import Path

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

    if results < .05 + .001:
        arr = np.array([True])
    return arr
def get_analysis(results = None, **kwargs):

    track = []
    html_ = ['<html style="display: flex; justify-content: center"><body style="background-color:#1a1a1a ;  color:white">']
    for x in results :
        for y in kwargs[ 'parameters']:
            kwargs['shift_parameter'] = int(y)
            kwargs['stock_list'] = x

            graph = kwargs['filter_func'](
        kwargs,True)
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

    Path('.html').write_text('\n'.join(html_), encoding='utf-8')




