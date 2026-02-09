import numpy as np
import pandas as pd, plotly.io as io
from optuna import Study
from plotly.subplots import make_subplots
import vectorbt as v, warnings
from Market_Analysis import get_time_period, get_yf
from multiprocessing import Pool
import optuna
from Filters import *

def make_objective(**kwargs):
    def objective(trial: optuna.trial.Trial ):

        tup= kwargs['tup']
        strat = kwargs['strat']
        z_threshold = trial.suggest_float('z_threshold', 1.1,  2.9 , step=0.2)
        print(tup)
        name = strat + '/' + str(tup) + '.parquet'
        roll = trial.suggest_int('roll', 20, 55, step= 5 )
        stck_list =  pd.read_parquet (name ) . index
        tup =  (tup [ 0] + 500, tup[1] +200 )

        results = runner_multiple(pd.DataFrame(index= [tuple(x) for x in  stck_list ]   ), [tup],
                        port_sim, type = 'validation',init_money=1000, strat_class = 'mv',
                        inputs=None,num_processes=16,   output_metrics=['Total Return', 'Sharpe', 'Alpha', 'Number of Trades'],
                        freq='d',graphs =  False, parameters_=[z_threshold,   roll ]  )

    # Optimization goal.
        results = results[[x for x in list(results.columns) if 'Alpha' in x]].mean(axis=1).mean()
        if results is np.nan:
            return 0
        return results
    return objective
#d
# Helper to execute the optimization
def parameter_optimization( tup,strat):

    name = strat + '/' + str(tup) + ''
    storage = 'sqlite:///opt_storage.db'
    stu = optuna.create_study(storage='sqlite:///opt_storage.db', direction='maximize',study_name= name ,load_if_exists=True)
    stu.optimize(make_objective(tup= tup , strat=strat),n_trials=5, gc_after_trial=True,catch=False,show_progress_bar = False)
    name = strat + '/' + str(tup) + 'optimize_results.parquet'
    stu.trials_dataframe().sort_values(by='value').to_parquet( name )
    print(pd.read_parquet( name ))

# For the multiprocessing.
def runner(stock_pair, shift_parameter , filter_func, **kwargs) -> pd.DataFrame | list:

    inputs = kwargs.pop ('inputs')
    args = [[None]] * len(stock_pair)
    processes = kwargs.pop('num_processes')
    if inputs:
        args = stock_pair[[x for x in stock_pair.columns if (str(shift_parameter) + inputs[0]) == x]]
        args = args.squeeze(1)
        args.index = [tuple(x) for x in args.index]

    p_list = [dict(stock_list=list(x), shift_parameter=shift_parameter, args=y) | kwargs for x, y in
              zip(stock_pair.index, args)]

    with Pool(processes=processes) as pool:
        filter_results = pool.map(filter_func, p_list)
#
    if kwargs['graphs']:
        return filter_results
    filter_results = np.array(filter_results)
    col = kwargs.pop('output_metrics')
    val = [str(shift_parameter) + ' ' + col[x] for x in range(len(col))]
    if filter_results.size == 0: return pd.DataFrame(index=stock_pair.index, columns=val)
    series_results = pd.DataFrame(index=stock_pair.index, data=filter_results, columns=val)
    series_results = series_results[series_results != 0].dropna(axis=0)

    return pd.concat([series_results, stock_pair], axis=1, join='outer').dropna()

# For organization purposes.  Gives a clean separation of time period parameters from the others.
def runner_multiple(stock_pair_list, back_test_periods, filter_func, **kwargs) -> pd.DataFrame:
    if type(back_test_periods[0]) ==  int:
        shift_parameter_list = []
        for x in range(back_test_periods[0]):
            if kwargs['type'] == 'training':
                tup = (500 *x, 500 *x  +500+ 0 )
            elif kwargs['type' ] == 'validation' :
                tup = (500 * x+ 500 ,  500 * x + 500   + 0 + 200)
            shift_parameter_list.append( tup )

    else:
        shift_parameter_list = back_test_periods
    if len(shift_parameter_list) == 1:
        return runner(stock_pair_list, shift_parameter_list[0], filter_func, **kwargs)
    return runner_multiple(runner(stock_pair_list, shift_parameter_list[0], filter_func, **kwargs),
                           shift_parameter_list[1:],
                           filter_func, **kwargs)

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
    rolling_obj_diff = stck_data .rolling(rolling)
    z_score = (stck_data  - rolling_obj_diff.mean()) / rolling_obj_diff.std()

    z_score = z_score.dropna()

    exits = ((z_score > z_threshold) & (z_score.shift(1) < z_threshold)) + 0
    entries = ((z_score < -1 * z_threshold) & (z_score.shift(1) > -1 * z_threshold)) + 0
    entries_exits =  (entries - exits) + 0
    entries_exits.columns = strat_param['stock_list']

    data_close = stck_data[strat_param['stock_list']].loc[entries_exits.index]

    sold_ideal = (1 / data_close * init_money ).astype(int)
    quantities_practical = entries_exits * sold_ideal
    quantities_practical = quantities_practical.replace(0, np.nan).ffill().fillna(0,)
    return quantities_practical

# Optional Visualization which returns a HTML formatted tuple/list of relevant graphs/values. Currently, includes
##a table of all trades,etc.
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
        ['Entry Timestamp', 'Exit Timestamp', 'Column', 'Direction', 'Return', 'Status']].sort_values(by='Entry Timestamp')
    positions['Column'] = [x[0] for x in positions ['Column']]

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
                                 title=str(shift_parameter  ) + ' time period. ' + str(strat_param['stock_list'])))
    fig = fig.update_traces(selector=dict(name='port'), line=dict(width=5))
    fig = fig.update_traces(selector=dict(name='SPY'), line=dict(width=5))
    fig = fig.update_layout(dict(xaxis=dict(title='Dates'))).update_layout(dict(yaxis=dict(title='Normalized Cumulative Returns')))
    fig_new = fig_new.update_layout(dict(width=200 * 5, height=300, xaxis=dict(title='%'), yaxis=dict(title='count'),
                                         title='Distribution of Port Returns')).update_traces(textfont=dict(size=8))
    fig_new = fig_new.update_layout(dict(plot_bgcolor='#1a1a1a', paper_bgcolor='#1a1a1a', font=dict(color='white'),
                                         legend=dict(bgcolor='#1a1a1a', bordercolor='white', title=dict(text='Legend'),
                                                     borderwidth=1, font=dict(color='white'))))

    html_list = [x.to_html(include_plotlyjs='cdn', include_mathjax=False, auto_play=False, full_html=False) for x in
                 [fig, fig_new]]

    return html_list[0], metrics_values.to_frame().to_html(), html_list[1], positions .rename(dict(Column='Stock', ),  axis = 1).to_html(index=False), pd.concat(
        [p.benchmark_returns().describe() , p.returns().describe()], axis=1).to_html(), metrics_values['Sharpe Ratio'],

# Does the actual portfolio simulation
def port_sim(strat_param, ):

    init_money = strat_param['init_money']
    stck_list = strat_param['stock_list']
    time_period = strat_param['shift_parameter']

    stck_data =    get_time_period(stck_list, time_peri=time_period)
    if strat_param['strat_class'].lower() == 'co-integration':quantities_practical = get_signals(strat_param, stck_data)
    elif strat_param['strat_class'] == 'mv':quantities_practical = get_signals_mv(strat_param, stck_data)

    data_close = stck_data[strat_param['stock_list']].loc[quantities_practical.index]

    benchmark_ret =  get_time_period(['SPY'], time_peri=time_period).loc[quantities_practical.index]
    benchmark_cum_returns = (1 + benchmark_ret).cumprod()

    p = v.Portfolio.from_orders(close=data_close, log=True, size=quantities_practical, size_type='TargetAmount',
                                init_cash=init_money, freq=strat_param['freq'], cash_sharing=True)
    metrics = [x for x in p.stats().index if 'Trade' not in x]
    metrics.remove('Benchmark Return [%]'), metrics.remove("Win Rate [%]")
    metrics_values = pd.concat(
        [p.stats()[metrics].to_frame(), p.returns_stats(benchmark_rets=benchmark_ret).iloc[-7:].to_frame()]).squeeze()

    cond =  metrics_values['Total Return [%]'] > 0 and metrics_values['Sharpe Ratio'] > 1.7

    show_graphs = strat_param['graphs']
    #
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

    # List of stocks which were tested. Obtained with extraction from the Yahoo finance with the condition that there are
    ##no NA values over the last 10 years. Length of data is 2515 points (last 10 years) and there are 46 6 stocks
    stck_list = pd.read_parquet ('Close10y1d.parquet')  . columns

    # Mean reversion is tested on the assumption these stocks returns series all mean revert
    #
    for x in range( 0,2 + 1 ):
        tup = (500 * x, 500 * x + 500 + 0)

        name = 'MV/Results' + str(tup) + '.parquet'
        runner_multiple(pd.DataFrame(index= [tuple([x]) for x in stck_list ]   ), [ tup ],
              port_sim, init_money=1000, type='training', strat_class = 'mv',
                        inputs=None,num_processes=16,   output_metrics=['Total Return', 'Sharpe', 'Alpha', 'Number of Trades'],
                        freq='d',graphs =  False, parameters_=[1.5,  26]). to_parquet (name)
        print(pd.read_parquet(name))

        parameter_optimization(tup,'MV')

    # Creates stock pairs which are unique
    stck_list = set(frozenset([x,y]) for x in stck_list for y in stck_list if x!=y)
    stck_list = list(stck_list)[:100]
    for x in range(0, 2 + 1):
        tup = (500 * x, 500 * x + 500 + 0)
        name = 'Co-integration/' + str(tup) + '.parquet'

        # Filters the given list of stock pairs
        runner_multiple(pd.DataFrame(index= [tuple( x ) for x in stck_list ]   ), [ tup ],
                        cointegration_filter, init_money=1000, strat_class = 'co-integration',
                        inputs=None,num_processes=16,   type = 'training', output_metrics=[''],
                        freq='d',graphs =  False, parameters_=[1.5,  26]). to_parquet (name)
        print(pd.read_parquet(name))

    #
        parameter_optimization(tup, 'Co-integration')

    # Visualization
    for x in range(0, 2 + 1):
        tup = (500 * x, 500 * x + 500 + 0)

        name = 'Co-integration/Results' + str(tup) + '.parquet'
        stck_list = pd.read_parquet(name) . index [ : ]
        tup = (tup [ 0 ] + 200 + 300, tup[ 1 ] +  255 )

        get_analysis(
    pd.DataFrame(index= [tuple( x ) for x in stck_list ]   ),   parameters = [ tup ] ,
                            filter_func= port_sim, init_money=1000, strat_class = 'co-integration',
                            inputs=None,num_processes=16,   type = 'training', output_metrics=[''],
                            freq='d',graphs =  True , parameters_=[1.5,  26])
    exit()
if __name__ == '__main__':
    run()
