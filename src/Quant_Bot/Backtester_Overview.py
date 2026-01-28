import numpy as np
import pandas as pd, plotly.graph_objs as g, plotly.io as io
import vectorbt
from plotly.subplots import make_subplots
import yfinance as yf, vectorbt as v, warnings
from Market_Analysis import full_stocks, get_time_period, get_yf
from statsmodels.tsa.stattools import adfuller, coint
from vectorbt.root_accessors import Vbt_DFAccessor
from multiprocessing import Pool, cpu_count
import optuna, statsmodels.api as m

warnings.filterwarnings('ignore', module='pd')
io.renderers.default = 'browser'

def cointegration_filter(cur_stock, show_graphs=False):
    cur_pair = get_time_period(cur_stock['stock_list'], True, freq=cur_stock['freq'],
                               num_data_points=cur_stock['num_p'], shift=int(cur_stock['shift_parameter']) + 1)
    cur_stock = cur_stock['stock_list']

    model = m.OLS((cur_pair[cur_stock[0]]), m.add_constant((cur_pair[cur_stock[1]]))).fit()
    results = coint(np.log(cur_pair[cur_stock[0]]), np.log(cur_pair[cur_stock[1]]))[1]
    if show_graphs:
        return model.resid.rolling(28).mean().vbt.plot(title=tuple(cur_stock[0:2]).__str__()).to_html(
            include_plotlyjs='cdn', include_mathjax=False, auto_play=False, full_html=False)
    arr = np.array([False])
    if results < .05 + .001 + 0:
        arr = np.array([True])
    return arr


def runner(stock_pair, shift_parameter: int, filter_func, **kwargs) -> pd.DataFrame:
    inputs = kwargs.pop('inputs')
    args = [[None]] * len(stock_pair)
    if inputs:
        args = stock_pair[[x for x in stock_pair.columns if (str(shift_parameter) + inputs[0]) == x]]
        args = args.squeeze(1)
        args.index = [tuple(x) for x in args.index]

    p_list = [dict(stock_list=list(x), shift_parameter=shift_parameter, args=y) | kwargs for x, y in
              zip(stock_pair.index, args)]
    with Pool(processes=15) as pool:
        filter_results = pool.map(filter_func, p_list)

    filter_results = np.array(filter_results)
    col = kwargs.pop('output_metrics')
    val = [str(shift_parameter) + ' ' + col[x] for x in range(len(col))]
    if filter_results.size == 0: return pd.DataFrame(index=stock_pair.index, columns=val)
    series_results = pd.DataFrame(index=stock_pair.index, data=filter_results, columns=val)
    series_results = series_results[series_results != 0].dropna(axis=0)

    return pd.concat([series_results, stock_pair], axis=1, join='outer').dropna()


def runner_multiple(stock_pair_list, shift_parameter_list, filter_func, **kwargs) -> pd.DataFrame:

    if len(shift_parameter_list) == 1:
        return runner(stock_pair_list, shift_parameter_list[0], filter_func, **kwargs)
    return runner_multiple(runner(stock_pair_list, shift_parameter_list[0], filter_func, **kwargs),
                           shift_parameter_list[1:],
                           filter_func, **kwargs)


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

    return entries_exits

def graphs_analysis(strat_param, **kwargs) -> tuple:
    shift_parameter = strat_param['shift_parameter']
    p = kwargs.pop('p')
    benchmark_cum_returns = kwargs.pop('benchmark_cum_returns')
    metrics_values = kwargs.pop('metrics_values')

    fig = make_subplots()
    cum_returns = (1 + p.close.pct_change()).cumprod()
    cum_returns.columns = strat_param['stock_list']
    cum_returns.vbt.plot(fig=fig)
    positions = p.positions.records_readable[
        ['Entry Timestamp', 'Exit Timestamp', 'Column', 'Direction', 'Return']].sort_values(by='Entry Timestamp')

    for x in strat_param['stock_list'][0:2]:
        positions_cur = positions[positions['Column'] == x].dropna()
        fig.add_scatter(x=positions_cur['Entry Timestamp'],
                        y=cum_returns[x].loc[[x for x in positions_cur['Entry Timestamp']]], name=x + ' Entry',
                        mode='markers', marker=dict(symbol='circle', size=10 / 2, color='yellow'),
                        hovertext=positions_cur['Direction'])
        fig.add_scatter(x=positions_cur['Exit Timestamp'],
                        y=cum_returns[x].loc[[x for x in positions_cur['Exit Timestamp']]], name=x + ' Exit',
                        mode='markers', marker=dict(symbol='circle', size=10 / 2, color='white'),
                        hovertext=positions_cur['Direction'])
    benchmark_cum_returns.vbt.plot(fig=fig)
    returns = (1 + p.returns()).cumprod().rename('port')
    returns.vbt.plot(fig=fig)
    fig_new = p.returns().rename('returns').vbt.histplot()
    rt = p.returns().describe().drop('count').rename()
    fig_new.add_scatter(x=list(rt), y=[0] * len(rt), mode='markers+text',
                        textposition=['top center', 'top center', 'top center', 'top center', 'bottom center',
                                      'top center', 'top center'], text=rt.index, name='')

    fig = fig.update_layout(dict(plot_bgcolor='#1a1a1a', paper_bgcolor='#1a1a1a', font=dict(color='white'),
                                 legend=dict(bgcolor='#1a1a1a', bordercolor='white', title=dict(text='Legend'),
                                             borderwidth=1, font=dict(color='white'))))
    fig = fig.update_layout(dict(width=1000, height=700,
                                 title=str(shift_parameter + 500) + ' time period. ' + str(tuple(strat_param[0:2]))))
    fig = fig.update_traces(selector=dict(name='port'), line=dict(width=5))
    fig = fig.update_traces(selector=dict(name='SPY'), line=dict(width=5))
    fig = fig.update_layout(dict(xaxis=dict(title='Dates'))).update_layout(dict(yaxis=dict(title='CummSum')))
    fig_new = fig_new.update_layout(dict(width=200 * 5, height=300, xaxis=dict(title='%'), yaxis=dict(title='count'),
                                         title='Distribution of PctReturns')).update_traces(textfont=dict(size=8))
    fig_new = fig_new.update_layout(dict(plot_bgcolor='#1a1a1a', paper_bgcolor='#1a1a1a', font=dict(color='white'),
                                         legend=dict(bgcolor='#1a1a1a', bordercolor='white', title=dict(text='Legend'),
                                                     borderwidth=1, font=dict(color='white'))))

    html_list = [x.to_html(include_plotlyjs='cdn', include_mathjax=False, auto_play=False, full_html=False) for x in
                 [fig, fig_new]]

    return html_list[0], metrics_values.to_frame().to_html(), html_list[1], p.positions.records_readable.sort_values(
        by='Entry Timestamp', ascending=True).drop(columns=['Position Id']).to_html(index=False), pd.concat(
        [0, p.returns().describe()], axis=1).to_html(), metrics_values['Sharpe Ratio'],

def port_sim(strat_param, show_graphs=False):

    init_money = strat_param['init_money']

    stck_data = get_time_period(strat_param['stock_list'] + ['SPY'], custom_data=True,
                                num_data_points=strat_param['num_p'], shift=strat_param['shift_parameter'] + 501,
                                freq=strat_param['freq'])
    entries_exits = get_signals(strat_param, stck_data)

    data_close = stck_data[strat_param['stock_list']].loc[entries_exits.index]

    sold_ideal = (1 / data_close * init_money).astype(int)
    quantities_practical = (entries_exits / entries_exits.abs()) * (
                sold_ideal * ((sold_ideal < entries_exits.abs()) + 0) + (
                    entries_exits.abs() * (entries_exits.abs() <= sold_ideal) + 0))

    benchmark_ret = stck_data['SPY'].pct_change()
    benchmark_cum_returns = (1 + benchmark_ret).cumprod()
    p = v.Portfolio.from_orders(close=data_close, log=True, size=quantities_practical, size_type='TargetAmount',
                                init_cash=init_money, freq=strat_param['freq'], cash_sharing=True)
    metrics = [x for x in p.stats().index if 'Trade' not in x]
    metrics.remove('Benchmark Return [%]'), metrics.remove("Win Rate [%]")
    metrics_values = pd.concat(
        [p.stats()[metrics].to_frame(), p.returns_stats(benchmark_rets=benchmark_ret).iloc[-7:].to_frame()]).squeeze()

    cond =  metrics_values['Total Return [%]'] > 0 and  metrics_values['Sharpe Ratio'] > 1.7 and metrics_values['Alpha'] > 1


    if cond:

        if not show_graphs:
            return [metrics_values[x] for x in metrics_values.keys() if any([y in x for y in strat_param['output_metrics']])] + [
                len(p.positions.records_readable)]
        return list(graphs_analysis(strat_param, p=p, benchmark_cum_returns=benchmark_cum_returns,
                                    metrics_values=metrics_values))

    else:
        if show_graphs:
            return None
        else:
            return np.zeros(len(strat_param['output_metrics']))

def run():

    pairs = pd.read_parquet('././Cointegration .parquet')

    pairs = pairs.index

    name = 'Cointegration_Results.parquet'
    runner_multiple(pd.DataFrame(index=[tuple(x) for x in pairs[10:12] if 'SPY' not in x]), [500], port_sim, init_money=1000,
                    inputs=None, num_p=400, output_metrics=['Total Return', 'Sharpe', 'Alpha', 'Num of Trades'],
                    freq='d', parameters_=[1.59, 25]).to_parquet(name)
    
    # z_threshold = 1.1


# #1.1, 4 5
if __name__ == '__main__':
    run()
