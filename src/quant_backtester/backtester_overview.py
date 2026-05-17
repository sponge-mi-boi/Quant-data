import itertools
from multiprocessing import Pool

import optuna

import vectorbt as v

from plotly.subplots import make_subplots

from .strategies import *
from .strategies import _get_signals, _get_signals_mv, _get_signals_mv_cross_asset, \
    _weights_alloc, _get_signals_momentum_tr, _get_signals_momentum_cross_asset


def runner_multiple(stock_pair_list, time_period_list: list , filter_func, **kwargs)   :
    """
    Separates execution by time period if multiple time periods are given as parameters.
        In the provided example, acts merely as an entry point before parallelization.
    Args:
        stock_pair_list: list
            Lists of lists which describe various assets universes which can be given non-zero weights during the execution of this strategy
        time_period_list: list
            A list of tuples of starting and ending position indices of testing periods

        filter_func:
            The function which is to be executed on all the assets in the portfolio. Usually, it is_port_sim.

        **kwargs:
            Strategy parameters
    Returns: pd.DataFrame
        Resulting metrics of the various portfolios which have been passed. If filtered, only those which pass the
        given filters are shown.

    """

    if type(stock_pair_list) == list: df_setup = pd.DataFrame(index=stock_pair_list)
    else: df_setup = stock_pair_list
    if df_setup.index.empty  : return pd.DataFrame()

    if len(time_period_list) == 1:
        return _runner(df_setup, time_period_list[0], filter_func, **kwargs)
    df_cur = _runner(df_setup, time_period_list[0], filter_func, **kwargs)

    return runner_multiple( df_cur ,
                           time_period_list[1:],
                           filter_func, **kwargs)


def _runner(stock_pair: pd.DataFrame, time_period, filter_func, **kwargs) -> pd.DataFrame :
    """

    Implements the multiprocessing. Executes the given filter_func on all given portfolios for a single time_period.

    Args:
        stock_pair:
        time_period:
        filter_func:
        **kwargs:

    Returns: pd.DataFrame | list
        If a graph is to be created, then an HTML formatted list of strings is returned. Otherwise, the resulting metrics
        of the various portfolios are returned, or an empty DataFrame is returned if the given filters are not passed.
    """
    inputs = kwargs.pop('inputs')
    args = [[None]] * len(stock_pair)
    processes = kwargs.pop('num_processes')

    parallel = kwargs.pop('parallel')
    if inputs:
        # args = stock_pair[[x for x in stock_pair.columns if (str(time_period) + inputs[0]) == x]]
        # args = args.squeeze(1)
        # args.index = [tuple(x) for x in args.index]
        return stock_pair

    p_list = [dict(stock_list=list(x), time_period=time_period, args=y) | kwargs for x, y in
              zip(stock_pair.index, args)]

    if not parallel :
        filter_results = []
        for list_ in p_list: filter_results.append(filter_func(list_))
    else:
        with Pool(processes=processes) as pool:
            filter_results = pool.map(filter_func, p_list)
    if kwargs['graphs']:
        return filter_results

    filter_results = np.array(filter_results)
    col = kwargs.pop('output_metrics')
    col = list(col.keys())

    val = np.array([str(time_period) + ' ' + col[x].replace('_', ' ') for x in range(len(col))])
    if filter_results.size == 0: return pd.DataFrame(index=stock_pair .index  , columns=val)

    series_results = pd.DataFrame(index=stock_pair . index  , data=filter_results, columns=val)
    return  pd.concat([series_results[series_results != 0].dropna(axis=0),stock_pair ],axis=1).dropna()

def _make_objective(**kwargs):
    """
    Helper function.

    Args:
        **kwargs:

    Returns:

    """
    kwargs_ = kwargs .copy()
    time_period = kwargs_.pop('time_period')

    stck_list = kwargs_.pop('stck_list')
    strat_classes = kwargs_.pop('strat_class')

    def objective(trial: optuna.trial.Trial):

        strat_cur = dict.fromkeys(strat_classes.keys())
        for x,y in strat_classes.items():
            strat = dict.fromkeys(y.keys())
            for k,z in y.items():
                if k == 'roll':
                    strat[k] = trial.suggest_int(k,z[0],z[1],step=z[2])
                elif k == 'z_threshold':
                    if type(z[0]) == tuple:
                        cur_thresh = []
                        for g in z:
                            cur_thresh.append(trial.suggest_float(k,g[0],g[1],step=g[2]))
                        strat[k] = tuple(cur_thresh)
                    else:
                        strat[k] = trial.suggest_float(k,z[0],z[1],step=z[2])
            strat_cur[x] = strat
        kwargs_['strat_class'] = strat_cur
        kwargs_['filter_func'] = _port_sim
        results = runner_multiple(pd.DataFrame(index=np.array([tuple(x) for x in stck_list])), [time_period],
                                  **kwargs_)

        results = results[[x for x in list(results.columns) if 'Alpha' in x]].mean(axis=1).mean()
        if results is np.nan:
            return 0
        return results


    return objective


def parameter_optimization(**kwargs,) :
    """
    Optimizes the parameters of the given strategy over the given assets and time period. Only the condition of maximizing
    Alpha of the resulting portfolio is given as the goal. Future versions will include more varied goals.

    Args:

    Returns:

    """
    type_ = kwargs.pop('type')

    if type_ == 'grid':

        time_period = kwargs .pop('time_period')

        strat_classes = kwargs .pop('strat_class')
        param = []
        for x, y in strat_classes.items():
            params = []
            keys = []
            roll_list = []
            z_list = []
            for k, z in y.items():
                keys.append(k)
                if k == 'z_threshold':
                    z_thresh_list = [x/10 for x in range( int(10*z[0] ), int(10*z[1] ),int(10*z[2]))]
                    z_list = z_thresh_list
                elif k == 'roll':
                    roll_list = list(range( z[0], z[1], z[2]   ))
            if not roll_list:
                for k in z_list:
                    st = dict.fromkeys([x])
                    strat = dict.fromkeys(y.keys())
                    strat[keys[0]] = k
                    st[x] = strat
                    params.append(st)
                param.append(params)
                continue
            for k in z_list:
                for z in roll_list:
                    st = dict.fromkeys([x])
                    strat = dict.fromkeys(y.keys())
                    strat[keys[0]] = k
                    strat[keys[1]] = z
                    st[x] = strat
                    params.append(st)
            param.append(params)
        strat_p = list(itertools.product(*param))


        best_alpha = 0
        best_st = dict()
        for x in strat_p:
            kwargs = kwargs.copy()
            kwargs_ = kwargs | dict(strat_class=x)
            results = runner_multiple(pd.DataFrame(index=np.array([tuple(x) for x in kwargs['stck_list']])), [time_period],
                                  **kwargs_)
            cur_alpha = results[[x for x in list(results.columns) if 'Alpha' in x]].mean(axis=1).mean()

            if cur_alpha and cur_alpha > best_alpha:
                best_alpha = cur_alpha
                best_st = x
        return best_st,best_alpha
    else:

        storage = 'Published/data/' + 'sqlite:///opt_storage.db'

        stu = optuna.create_study(storage=storage, direction='maximize', study_name='',
                                  load_if_exists=True)
        stu.optimize(_make_objective(**kwargs), n_trials=1, gc_after_trial=True,

                     show_progress_bar=False)

        stu.trials_dataframe().sort_values(by='value')
    return stu.best_params,stu.best_value

def graphs_analysis(strat_param, **kwargs) -> tuple:

    """
    Optional Visualization which returns an HTML formatted tuple of relevant graphs/values. Used with_port_sim

    Args:
        strat_param:
        **kwargs:

    Returns:

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


def _port_sim(strat_param, ) -> np.ndarray | list | None:
    """
    Portfolio simulation for a single portfolio.

    Note:
        Uses vectorbt.
    Args:
        strat_param:


    Returns: list | np.array
        Returns a list if graphs are to created, otherwise returns the results of the strategy, if they pass the given
        filters, as a np.array. If they do not pass, an array of zeros is returned.

    """

    init_money = strat_param['init_money']
    stck_list = strat_param['stock_list']
    unique_stks = list(dict.fromkeys(stck_list))

    time_period = strat_param['time_period']
    stck_data = get_time_period(unique_stks, time_peri=time_period)
    raw_entries_exits_list = dict()

    for strat in strat_param['strat_class'].keys():
        cur_param = strat_param.copy()
        cur_param['strat_class'] = strat
        cur_param['parameters_'] = strat_param['strat_class'][strat]
        if strat.lower() == 'cointegration':
            ent = (_get_signals(cur_param, stck_data))  [unique_stks]
            strat_param['weights_filter']['cointegration']['data'] = ent.copy()
            ent = ent/ent.abs()

            raw_entries_exits_list[strat] = ent

        elif strat.lower() == 'mv':

            raw_entries_exits_list[strat] = (_get_signals_mv(cur_param, stck_data))[unique_stks]

        elif strat.lower() == 'cross_asset_mv':

            raw_entries_exits_list[strat] = (_get_signals_mv_cross_asset(cur_param, ))[unique_stks]

        elif strat.lower() == 'momentum_trending':
            raw_entries_exits_list[strat] = (_get_signals_momentum_tr(cur_param, stck_data))[unique_stks]
        elif strat.lower() == 'cross_asset_momentum_trending':
            raw_entries_exits_list[strat] = _get_signals_momentum_cross_asset(cur_param)[unique_stks]

    if 'regime_estimator' in strat_param['weights_filter'].keys():
        regime_estimator_ = strat_param['weights_filter']['regime_estimator']
    else:
        regime_estimator_ = False
    dta = list(raw_entries_exits_list.values())[0]

    if regime_estimator_:

        market_states = regime_estimator(strat_param)

        raw_entries_exits = 0

        for x, y in raw_entries_exits_list.items():

            raw_entries_exits +=y.multiply(market_states[x], axis=0)


        raw_entries_exits = raw_entries_exits.dropna()

    else:
        raw_entries_exits = dta

    filter_list = strat_param['weights_filter'].copy()

    entries_exits = _weights_alloc(strat_param, raw_entries_exits, filter_list)

    data_close = stck_data[unique_stks].loc[entries_exits.index]
    quantities_practical = (entries_exits * init_money).div(data_close, fill_value=0).astype(int)
    benchmark_ret = get_time_period(['SPY'], time_peri=time_period).loc[
        quantities_practical.index].pct_change().squeeze()

    benchmark_cum_returns = (1 + benchmark_ret).cumprod().squeeze()

    p = v.Portfolio.from_orders(close=data_close, size=quantities_practical, size_type='TargetAmount',
                                init_cash=init_money, freq=strat_param['freq'], cash_sharing=True)

    metrics = [x for x in p.stats().index if 'Trade' not in x]

    metrics_values = pd.concat(
        [p.stats()[metrics].to_frame(), p.returns_stats(benchmark_rets=benchmark_ret).iloc[-7:].to_frame()]).squeeze()

    output_metrics = strat_param['output_metrics']
    outputs_cond = pd.Series()

    for x, y in output_metrics.items():

        m = x.replace('_', ' ')
        if 'number of trades' == m.lower():
            if not y:
                outputs_cond[m] = True, np.float64(len(p.positions.records_readable))
            else:
                outputs_cond[m] = len(p.positions.records_readable) > y, np.float64(len(
                p.positions.records_readable) )
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
