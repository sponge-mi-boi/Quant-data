from multiprocessing import Pool

import optuna
import vectorbt as v
from plotly.subplots import make_subplots

from .strategies import _get_signals, _get_signals_mv, _get_signals_mv_cross_asset, \
    _weights_alloc, _get_signals_momentum_tr
from .strategies import *


def runner_multiple(stock_pair_list, back_test_periods, filter_func, **kwargs) -> pd.DataFrame:
    """
    For organization purposes. Gives a clean separation of time period parameters from the others. Simulates trading of
    various possible portfolios defined by a list of lists of possible assets.
        :param stock_pair_list: list
            Lists of lists which describe various assets universes which can be given non-zero weights during the execution of this strategy
        :param back_test_periods: list
            A list of tuples of starting and ending position indices of testing periods
        :param filter_func: function
            The function which is to be executed on all the assets in the portfolio. Usually, it is_port_sim.
        :param kwargs:
        :return: pd.DataFrame
            Resulting metrics of the various portfolios which have been passed. If filtered, only those which pass the
            given filters are shown.
    """
    if not stock_pair_list.index[0]: return pd.DataFrame()
    if type(back_test_periods[0]) == int:
        time_period_list = []
        for x in range(back_test_periods[0]):
            if kwargs['type'] == 'training':
                time_period = (500 * x, 500 * x + 500 + 0)
            elif kwargs['type'] == 'validation':
                time_period = (500 * x + 500, 500 * x + 500 + 0 + 200)
            time_period_list.append(time_period)

    else:
        time_period_list = back_test_periods
    if len(time_period_list) == 1:
        return _runner(stock_pair_list, time_period_list[0], filter_func, **kwargs)
    return runner_multiple(_runner(stock_pair_list, time_period_list[0], filter_func, **kwargs),
                           time_period_list[1:],
                           filter_func, **kwargs)


def _runner(stock_pair, time_period, filter_func, **kwargs) -> pd.DataFrame | list:
    """
    Implements the multiprocessing. Executes the given filter_func on all given portfolios for a single time_period.
    :param stock_pair: list
    :param time_period: tuple
    :param filter_func: function
    :param kwargs:
    :return: pd.DataFrame | list
        If a graph is to be created, then an HTML formatted list of strings is returned. Otherwise, the resulting metrics
        of the various portfolios are returned, or an empty DataFrame is returned if the given filters are not passed.
    """
    inputs = kwargs.pop('inputs')
    args = [[None]] * len(stock_pair)
    processes = kwargs.pop('num_processes')
    if inputs:
        args = stock_pair[[x for x in stock_pair.columns if (str(time_period) + inputs[0]) == x]]
        args = args.squeeze(1)
        args.index = [tuple(x) for x in args.index]

    p_list = [dict(stock_list=list(x), time_period=time_period, args=y) | kwargs for x, y in
              zip(stock_pair.index, args)]

    with Pool(processes=processes) as pool:
        filter_results = pool.map(filter_func, p_list)

    if kwargs['graphs']:
        return filter_results

    filter_results = np.array(filter_results)
    col = kwargs.pop('output_metrics')
    col = list(col.keys())

    val = [str(time_period) + ' ' + col[x].replace('_', ' ') for x in range(len(col))]
    if filter_results.size == 0: return pd.DataFrame(index=stock_pair.index, columns=val)
    series_results = pd.DataFrame(index=stock_pair.index, data=filter_results, columns=val)
    series_results = series_results[series_results != 0].dropna(axis=0)

    return pd.concat([series_results, stock_pair], axis=1, join='outer').dropna()


def _make_objective(**kwargs):
    """
    Optuna helper function.
    :param kwargs:
    :return:
    """
    def objective(trial: optuna.trial.Trial):
        time_period = kwargs['time_period']
        strat = kwargs['strat']
        z_threshold = trial.suggest_float('z_threshold', 1.1, 2.5, step=0.2 - .1)
        path = Path(__file__).parents[2]
        name = str(path) + '/artifacts/' + strat + '/results_training_' + str(time_period) + '.parquet'

        roll = trial.suggest_int('roll', 20, 55, step=5)
        stck_list = pd.read_parquet(name).index
        time_period = (time_period[0] + 500, time_period[1] + 200)

        results = runner_multiple(pd.DataFrame(index=[tuple(x) for x in stck_list]), [time_period],
                                 _port_sim, init_money=1000, strat_class=strat,
                                  inputs=None, num_processes=16,
                                  output_metrics=dict(Total_Return=0, Sharpe=1.7, Alpha=1 - .9, Number_of_Trades=2 / 3),
                                  freq='d', graphs=False, parameters_=dict(z_threshold=z_threshold, roll=roll))
        # Optimization goal.

        results = results[[x for x in list(results.columns) if 'Number of Trades' in x]].mean(axis=1).mean()
        if results is np.nan:
            return 0
        return results

    return objective


def parameter_optimization(strat_param, type_='Bayesian') -> None:
    """
    Optimizes the parameters of the given strategy over the given assets and time period. Only the condition of maximizing
    Alpha of the resulting portfolio is given as the goal. Future versions will include more varied optimization goals possibilities.
    :param strat_param: dict
    :param type_: str, default is 'Bayesian'
        The type of optimization method to use in the optimization. Only, Bayesian and grid are currently supported.
    :return: None
    """
    strat = strat_param['strat_class']
    time_period = strat_param['time_period']
    stck_list = strat_param['stock_list']
    path = Path(__file__).parents[2]
    name = str(path) + '/artifacts/' + strat + '/optimize_results/' + str(time_period) + '_' + '.parquet'

    roll_list = np.linspace(11, int(len(time_period) / 2), 10)
    z_th_list = np.linspace(1.5, 2.0, 5)
    best_alpha = 0
    if type_ == 'grid':
        for z_threshold in z_th_list:
            for roll in roll_list:
                roll = int(roll)
                runner_multiple(pd.DataFrame(index=[tuple([x]) for x in stck_list]), [time_period],
                               _port_sim, init_money=1000, strat_class=strat,
                                inputs=None, num_processes=16,
                                output_metrics=dict(Total_Return=0, Sharpe=1.7, Alpha=False, Number_of_Trades=False),
                                freq='d', graphs=False,
                                parameters_=dict(z_threshold=z_threshold, roll=roll)).to_parquet(name)
                cur_alpha = pd.read_parquet(name).sort_values(by=str(time_period) + ' Alpha').sum()
                if cur_alpha > best_alpha:
                    best_alpha= cur_alpha
                    print(pd.read_parquet(name).sort_values(by=str(time_period) + ' Alpha')   )

    else:
        name = strat + '/' + str(time_period) + ''
        storage = 'sqlite:///opt_storage.db'

        stu = optuna.create_study(storage=storage, direction='maximize', study_name=name,
                                  load_if_exists=True)
        stu.optimize(_make_objective(time_period=time_period, strat=strat), n_trials=5, gc_after_trial=True, catch=False,
                     show_progress_bar=False)
        path = Path(__file__).parents[2]
        name = str(path) + '/artifacts/' + strat + '/optimize_results/' + str(time_period) + '.parquet'

        stu.trials_dataframe().sort_values(by='value').to_parquet(name)
        print(pd.read_parquet(name).sort_values(by='value'))



def graphs_analysis(strat_param, **kwargs) -> tuple:
    """
    Optional Visualization which returns an HTML formatted tuple  of relevant graphs/values. Used with_port_sim
    :param strat_param: dict
    :param kwargs:
    :return: tuple
    """
    time_period = strat_param['time_period']
    p = kwargs.pop('p')
    benchmark_cum_returns = kwargs.pop('benchmark_cum_returns')
    metrics_values = kwargs.pop('metrics_values')

    fig = make_subplots()
    cum_returns = (1 + p.close.pct_change()).cumprod()
    cum_returns.columns = strat_param['stock_list']
    cum_returns.vbt.plot(fig=fig)
    positions = p.positions.records_readable[
        ['Entry Timestamp', 'Exit Timestamp', 'Column', 'Direction', 'Return', 'Status']].sort_values(
        by='Entry Timestamp')
    positions['Column'] = [x[0] for x in positions['Column']]

    for x in strat_param['stock_list']:
        positions_cur = positions[positions['Column'] == x].dropna()
        fig.add_scatter(x=positions_cur['Entry Timestamp'],
                        y=cum_returns[x].loc[[x for x in positions_cur['Entry Timestamp']]], name=x + ' Entry',
                        mode='markers', marker=dict(symbol='circle', size=10 / 2, color='yellow'),
                        hovertext=positions_cur['Direction'])
        fig.add_scatter(x=positions_cur['Exit Timestamp'],
                        y=cum_returns[x].loc[[x for x in positions_cur['Exit Timestamp']]], name=x + ' Exit',
                        mode='markers', marker=dict(symbol='circle', size=10 / 2, color='purple'),
                        hovertext=positions_cur['Direction'])

    benchmark_cum_returns.vbt.plot(fig=fig)
    returns = (1 + p.returns()).cumprod().rename('port')
    returns.vbt.plot(fig=fig)
    fig_new = p.returns().rename('returns').vbt.histplot()
    rt = p.returns().describe().drop('count').rename()
    fig_new.add_scatter(x=list(rt), y=[0] * len(rt), mode='markers+text',
                        textposition=['top center', 'top center', 'bottom center', 'top center', 'bottom center',
                                      'top center', 'top center'], text=rt.index, name='')

    fig = fig.update_layout(dict(plot_bgcolor='#1a1a1a', paper_bgcolor='#1a1a1a', font=dict(color='white'),
                                 legend=dict(bgcolor='#1a1a1a', bordercolor='white', title=dict(text='Legend'),
                                             borderwidth=1, font=dict(color='white'))))
    fig = fig.update_layout(dict(width=1000, height=700,
                                 title=str(time_period) + ' time period. ' + str(strat_param['stock_list'])))
    fig = fig.update_traces(selector=dict(name='port'), line=dict(width=5))
    fig = fig.update_traces(selector=dict(name='SPY'), line=dict(width=5))
    fig = fig.update_layout(dict(xaxis=dict(title='Dates'))).update_layout(
        dict(yaxis=dict(title='Normalized Cumulative Returns')))
    fig_new = fig_new.update_layout(dict(width=200 * 5, height=300, xaxis=dict(title='%'), yaxis=dict(title='Count'),
                                         title='Distribution of Port Returns')).update_traces(textfont=dict(size=8))
    fig_new = fig_new.update_layout(dict(plot_bgcolor='#1a1a1a', paper_bgcolor='#1a1a1a', font=dict(color='white'),
                                         legend=dict(bgcolor='#1a1a1a', bordercolor='white', title=dict(text='Legend'),
                                                     borderwidth=1, font=dict(color='white'))))

    html_list = [x.to_html(include_plotlyjs='cdn', include_mathjax=False, auto_play=False, full_html=False) for x in
                 [fig, fig_new]]

    return html_list[0], metrics_values.to_frame().to_html(), html_list[1], positions.rename(dict(Column='Stock', ),
                                                                                             axis=1).to_html(
        index=False),


def _port_sim(strat_param, ):
    """
    Portfolio simulation. Executed with the help of vectorbt.
    :param strat_param: dict
    :return: list | np.array
        Returns a list if graphs are to created, otherwise returns the results of the strategy, if they pass the given
        filters, as a np.array. If they do not pass, an array of zeros is returned.
    """
    init_money = strat_param['init_money']
    stck_list = strat_param['stock_list']
    unique_stks = list (set(stck_list) )
    time_period = strat_param['time_period']

    stck_data = get_time_period(unique_stks, True, time_peri=time_period)
    if strat_param['strat_class'].lower() == 'cointegration':
        raw_entries_exits = _get_signals(strat_param, stck_data)

    elif strat_param['strat_class'].lower() == 'mv':
        raw_entries_exits = _get_signals_mv(strat_param, stck_data)

    elif strat_param['strat_class'].lower() == 'cross_asset_mv':
        raw_entries_exits = _get_signals_mv_cross_asset(strat_param, )

    elif strat_param['strat_class'] == 'momentum_trending':
        raw_entries_exits = _get_signals_momentum_tr(strat_param, stck_data)

    entries_exits = _weights_alloc(strat_param, raw_entries_exits, strat_param['weights_filter'])

    data_close = stck_data[unique_stks].loc[entries_exits.index]
    quantities_practical = (entries_exits * init_money).div(data_close, fill_value=0).astype(int)
    benchmark_ret = get_time_period(['SPY'], True, time_peri=time_period).loc[
        quantities_practical.index].pct_change().squeeze()

    benchmark_cum_returns = (1 + benchmark_ret).cumprod().squeeze()

    p = v.Portfolio.from_orders(close=data_close, log=True, size=quantities_practical, size_type='TargetAmount',
                                init_cash=init_money, freq=strat_param['freq'], cash_sharing=True)
    metrics = [x for x in p.stats().index if 'Trade' not in x]

    metrics_values = pd.concat(
        [p.stats()[metrics].to_frame(), p.returns_stats(benchmark_rets=benchmark_ret).iloc[-7:].to_frame()]).squeeze()

    output_metrics = strat_param['output_metrics']
    outputs_cond = pd.Series()
    for x, y in output_metrics.items():

        m = x.replace('_', ' ')
        if 'number of trades' == m.lower(): outputs_cond[m] = len(p.positions.records_readable) > y, len(
            p.positions.records_readable)
        for a in metrics_values.keys():
            if m.lower() in a.lower():
                if not y:
                    outputs_cond[m] = True, metrics_values[a]
                else:
                    outputs_cond[m] = metrics_values[a] > y, metrics_values[a]
    cond = all([x[0] for x in outputs_cond])
    show_graphs = strat_param['graphs']

    if cond:

        if not show_graphs:
            return [x[1] for x in outputs_cond]
        return list(graphs_analysis(strat_param, p=p, benchmark_cum_returns=benchmark_cum_returns,
                                    metrics_values=metrics_values))

    else:
        if show_graphs:
            return None
        else:
            return np.zeros(len(strat_param['output_metrics']))
