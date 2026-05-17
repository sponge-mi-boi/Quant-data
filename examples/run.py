"""
This module demonstrates the use of the package quant_backtester.

    - The primary method is run() and a generalized example is shown in example.
    - Results are stored in parquet files, as dataframes, with each horizontal representing a portfolio.
    - The index is the assets in that portfolio, and the columns are defining possible metrics of evaluating the performance.
    - There is also a focus on efficient storage and collection of all results with SQL.


For convention purposes and to define the backtesting framework, the following definitions are given.

There are three time periods for each backtest.
    - Training period
        - Primarily to determine asset behavior or to restrict the size of possible tradable assets.
    - Validation period
       - Primarily to test possible different combinations of the selected assets. Another possible use is parameter optimization.
    - Testing period
       - Performance metrics upon a chosen combination is used to determine if the strategy was successful.

Each period has corresponding aspects which define the portfolio being tested for that period.

The set of available assets for each period are defined as follows.
    - u
        - The set of assets available for trade in the training period.
    - n
        - The set of assets available for trade in the validation period.
    - k
        - The set of assets available for trade in the testing period.

The set of available strategies, filters, and performance metrics to use for each period are also subject to change.

The set of all available strategies are as follows. See strategies.py for further details

    - Cointegration time based

    - Mean reversion time based

    - Mean reversion cross asset

    - Momentum trending time based

The set of all available filters are as follows. Filters are classified into four types, each having a different use case.
See market_filters_analysis for further details.

    - Pre-training filters
        - These are used to reduce the size of the set of possible tradeable assets, before the backtesting process even begins.
        - The available ones are as follows.

           - Market cap

           - Correlation

           - Volatility

           - Cointegration

           - Mean reversion cross asset

    - Period filters
        - These are used to assess performance of particular portfolios and filter based on custom principle
        - Included ones are based on metrics Total Return, Sharpe Ratio, Alpha, Number of Trades, and combinations of them
    - Market state filters
        - These are used to determine which strategies to use based on market state variables
    Rolling filters
        - These are used to determine which assets to assign non-zero weights or to give ranked weights executed for
          defined frequency of time steps
        - The available ones are as follows.
            highest z_score, lowest standard deviation, highest mean, most positive mean count
    - Weight filters
       - These are used to help assign weights depending on the defined aspects of a portfolio
       - The available ones are as follows.
            - pc(n) neutrality
            - beta neutrality
            - dollar neutrality

Note the portfolio is assumed to be always fully deployed for backtesting purposes.
"""

import json
import warnings
from itertools import combinations
from pathlib import Path

import duckdb
import numpy as np
import optuna
import pandas as pd

import quant_backtester
from quant_backtester import *

warnings.filterwarnings('ignore')
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.width', N
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', lambda x: str(round(float(x), 4)))
pd.set_option('display.precision', 4)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

np.set_printoptions(suppress=True)
optuna.logging.set_verbosity(optuna.logging.ERROR)


def _database_create(path, new_database=False):
    """

    Create a new database or loads an existing one

    Args:
        path: str
            Path to the database
        new_database: bool, default is False
            Create a new database to reset

    Returns: DuckDBPyConnection
            The cursor of the database
    """
    cursor = duckdb.connect(path + '/artifacts/results_storage.db').cursor()
    tables = cursor.execute('SHOW TABLES').fetchall()
    tables.sort(key=lambda table: '_' not in table[0])
    if new_database:

        with open(path + '/artifacts/store_results_v.json', 'w') as f:
            json.dump([], f, indent=4)
        for table in tables:
            cmd_del = 'DELETE FROM ' + table[0]
            cursor.execute(cmd_del)
            cmd_del = 'DROP TABLE   ' + table[0]
            cursor.execute(cmd_del)
        files = Path(str(path) + '/artifacts/files').iterdir()
        for result in files: result.unlink()

        with open(str(path) + '/artifacts/outline.sql') as file:
            txt = file.read()[3:]
        cursor.execute(txt)
        exit()
    return cursor


def _database_update(cursor, current_state, ):
    """
    Updates the database with the current back test that was completed
    Args:
        cursor: DuckDBPyConnection
        current_state: dict
            The state of the current test
    Returns: None
    """
    strat_classes = current_state['strategies']
    name = current_state['name']
    time_period = current_state['time_period']
    type_ = current_state['type']
    previous_id = current_state['reference_id']
    previous_id_ = current_state['reference_id_']
    filters = list(current_state['filters'].keys())

    for x in strat_classes:
        strategy_id_length = _col_length_database(cursor, 'strategies')
        strategy_id = cursor.execute('SELECT id FROM strategies WHERE name = ? ', [x]).fetchall()

        if not strategy_id:
            cursor.execute('INSERT OR IGNORE INTO strategies (id,name) VALUES (?,?)'
                           , [strategy_id_length, x])
        else:
            strategy_id_length = strategy_id[0][0]

        for y, d in strat_classes[x].items():
            parameters_id_length = _col_length_database(cursor, 'parameters')
            parameters_id = \
                cursor.execute('SELECT id FROM parameters WHERE(name , value  )  = (?, ?  )', [y, d]).fetchall()
            if not parameters_id:
                cursor.execute('INSERT OR IGNORE INTO parameters (id,name,value) VALUES (?,?,?)'
                               , [parameters_id_length, y, d])
            else:
                parameters_id_length = parameters_id[0][0]

            cursor.execute(
                'INSERT OR IGNORE INTO strategies_parameters (strategy_id,parameter_id) VALUES (?,?  )',
                [strategy_id_length, parameters_id_length])

    for x in filters:
        filters_id = _col_length_database(cursor, 'filters')
        cursor.execute('INSERT OR IGNORE INTO filters (id,name) VALUES (?,?)'
                       , [filters_id, x])

    cursor.commit()

    results_id = cursor.execute('SELECT id FROM results  WHERE name = ? ', [name]).fetchall()

    if not results_id:
        results_id = _col_length_database(cursor, 'results')
        cursor.execute(
            'INSERT OR IGNORE INTO results(id,name,time_period,type,reference_id,reference_id_  ) VALUES (?,?,?,?,?,?)',
            [results_id, name, time_period, type_, previous_id, previous_id_])

    else:
        results_id = results_id[0][0]

    for strat in strat_classes.keys():
        strategy_id = cursor.execute('SELECT id FROM strategies WHERE name = ? ', [strat]).fetchall()[0][0]

        for x, y in strat_classes[strat].items():
            parameter_id = \
                cursor.execute('SELECT id FROM parameters WHERE(name , value  )  = (?, ?  )', [x, y]).fetchall()[0][0]

            for filter in filters:
                filters_id = cursor.execute('SELECT id FROM filters WHERE name = ? ', [filter]).fetchall()[0][0]

                cursor.execute(
                    'INSERT OR IGNORE INTO file_labels (file_id,strategy_id,filters_id,parameter_id) VALUES (?,?,?, ?   )',
                    [results_id, strategy_id, filters_id, parameter_id])


def _get_state_database(cursor, reset=False):
    """
    Prints all the relationships stored in the database
    Args:
        cursor: DuckDBPyConnection
        reset: bool, default is False
            If the database is to be reset
    Returns: None
    """
    tables = cursor.execute('SHOW TABLES').fetchall()
    tables.sort(key=lambda table: '_' not in table[0])
    print('-------------------------' * 10)
    for table in tables:

        if reset:
            cmd_del = 'DELETE FROM ' + table[0]
            cursor.execute(cmd_del)
            cmd_del = 'DROP TABLE   ' + table[0]
            cursor.execute(cmd_del)
            continue
        cmd = 'SELECT * FROM ' + table[0]
        print('\n' + table[0])
        print(cursor.execute(cmd).df())
    print('-------------------------' * 10)
    print(pd.DataFrame (_get_relationships(cursor)))


def _col_length_database(cursor, table_name):
    """
    Wrapper to obtain the number of entries stored in a table
    Args:
        cursor: DuckDBPyConnection
        table_name: str
            The length of the table to obtain
    Returns: int
        The length of the table
    """
    cmd = 'SELECT * FROM ' + table_name
    return len(cursor.execute(cmd).fetchall())


def _get_relationships(cursor):
    """
    Returns all the relational data stored in the database
    Args:
        cursor:  DuckDBPyConnection
    Returns: dict
        A dictionary representation of the relationships stored in the database
    """
    results_df = cursor.execute('''

WITH 
    h_2 AS (
    SELECT results.name, LIST(DISTINCT results.type),LIST(DISTINCT results.time_period),LIST(DISTINCT(strategies.name, (parameters.name,parameters.value ))),  LIST(DISTINCT filters.name), LIST(DISTINCT reference_id), LIST(DISTINCT reference_id_)
    FROM results, strategies, filters,  parameters
    JOIN strategies_parameters ON (strategy_id,parameter_id) = (strategies.id,parameters.id)
    JOIN file_labels ON (file_labels.strategy_id, file_labels.parameter_id, file_id,  file_labels.filters_id) =( strategies_parameters.strategy_id, strategies_parameters.parameter_id, results.id, filters.id)
    GROUP BY results.name 
    )
SELECT * 
FROM h_2   
    
    ''').fetchall()
    
    if not results_df:
        keys = ['time_period', 'name', 'type', 'strategies', 'filters', 'parameters']
        return [dict.fromkeys(keys)]

    dict_rep = list()

    for results in results_df:
        results = list(results)
        cur = dict()

        result = results
        cur['name'] = result.pop(0)
        cur['time_period'] = result.pop(1)[0]

        cur['type'] = result.pop(0)[0]
        cur['reference_id'] = result.pop(-2)[0]
        cur['reference_id_'] = result.pop(-1)[0]

        cur['filters'] = result.pop(-1)
        result_current = result.pop(0)

        strategies = dict()
        for x in result_current:
            if x[0] not in strategies.keys():
                strategies[x[0]] =  dict([x[1]])
            else:
                strategies[x[0]] |= dict([x[1]])

        for x in strategies.keys():
            for y in strategies[x] .keys(): strategies[x][y] = round(strategies[x][y],2)

        cur['strategies'] = strategies

        dict_rep.append(cur)

    return dict_rep


def training(current_state, u , n_count=10, num_processes=16, metrics=None, initial_filter_period=None, visualize=False,
             executed=False, parallel=True):
    """
    Executes the simulation of singleton portfolios based on the inputted assets. Note the primary use of this period is
    to determine asset behavior as currently constructed.

    Args:
        parallel: bool, default is True
        current_state: dict
            The state of the current test
        u: list
            The assets which can be traded
        n_count: int, default is 10
            The size of the resulting set of possible assets n
        num_processes: int, default is 16
            The number of parallel processes to use upon portfolio trading simulation
        metrics: dict, default is None
            The metric filter to use to assess portfolio performance
        initial_filter_period: dict, default is None
            If there is a separate filter to be used based on different asset qualities before portfolio simulation
        visualize: bool, default is False
            Stores an HTML rendered graph output of the portfolio simulation
        executed: bool, default is False
            If the current portfolio test described by current_state has already been executed

    Returns: list
        The filtered assets
    """
    time_period = current_state['time_period']
    # Filter period to reduce the size of this universe

    if initial_filter_period:
        period_length = initial_filter_period['filter_initial']
        fp = (time_period[0], time_period[0] + period_length)
        for name in initial_filter_period.keys():
            if 'market_cap' == name:
                parameters = initial_filter_period[name]
                u = market_cap_filter(dict(stock_list=u, time_period=fp, freq='d'), parameters['type_'])

            elif 'correlation' == name:
                parameters = initial_filter_period[name]
                u = corr_filter(dict(stock_list=u, time_period=fp, freq='d'), parameters['c_threshold'])

            elif 'cointegration' == name:


                path = Path(__file__).parents[1]

                path = str(path) + '/data/processed/'

                if Path.exists(Path(path + 'cointegration_'+str(time_period) + '.parquet')):
                    u = pd.read_parquet(path + 'cointegration_'+str(time_period) + '.parquet').index
                    u = list(u)
                else:
                    u_temp = list(dict.fromkeys(
                        tuple(dict.fromkeys([x, y])) for x in u for y in u if x != y and 'SPY' not in [x, y]))

                    u = runner_multiple(list(u_temp), [fp],
                                        cointegration_filter, freq='d', strat_class=None,
                                        inputs=None, parallel=parallel, graphs=False, output_metrics=dict(p=.05),
                                        num_processes=num_processes, )
                    u.to_parquet(path + 'cointegration_'+str(time_period) + '.parquet')

                    u = list(u.index)
                u = [y  for x in u for y in x]
            elif 'volatility' == name :
                parameters = initial_filter_period[name]
                u = volatility_filter(dict(stock_list=u, time_period=fp, freq='d'), parameters['q_threshold'])

        testing_period = (fp[1], time_period[1])
    else:
        testing_period = time_period
    strat_classes = current_state['strategies']

    name = current_state['name']
    filters = current_state['filters'].copy()
    if 'none' in filters.keys():
        filters.pop('none')
    if executed: return [tuple(x) for x in
                         pd.read_parquet(current_state['name']).sort_values(by=str(testing_period) + ' Alpha').index[
                             -n_count:]]


    if 'cointegration' in strat_classes.keys() and 'cointegration' not in initial_filter_period.keys():
        u = dict.fromkeys(tuple(dict.fromkeys([x, y])) for x in u for y in u if x != y and 'SPY' not in [x, y])
        u = list(u)
        # u =[tuple(x for y in u for x in y)]

    if not metrics: metrics = dict(Total_Return=   0.0001 , Sharpe=False, Alpha=False, Number_of_Trades=False)

    runner_multiple( [tuple([x]) for x in u[:]], [testing_period],
                    _port_sim, init_money=1000, strat_class=strat_classes,
                    inputs=None, num_processes=16,
                    output_metrics=metrics,
                    freq='d', weights_filter=filters  , graphs=False, parallel=  parallel
                    ).to_parquet(
        name)

    n = pd.read_parquet(name).sort_values(by=str(testing_period) + ' Sharpe').index[-n_count:]

    if visualize:
        get_analysis(
            [tuple(u)], parameters=[time_period],
            filter_func=_port_sim, init_money=1000, strat_class=strat_classes,
            inputs=None, num_processes=1,
            output_metrics=dict(),
            freq='d', weights_filter=None, graphs=True, )
    n = [tuple(x) for x in n]
    return n


def validating(n, current_state, type_=None, visualize=False, m_two=None, executed=False, num_processes=18,optimize=False, parallel=True):
    """
        Executes the simulation of singleton portfolios based on the inputted assets. Note the primary use of this period is
    to determine optimal parameters and to test all possible different combinations of assets.

    Args:
        type_:
        optimize:
        num_processes:
            The number of parallel processes to use upon portfolio trading simulation
        parallel: bool, default is True
        n: list
            The assets which can be traded in the validation period
        current_state: dict
        visualize: bool, default is False
            Stores an HTML rendered graph output of the portfolio simulation
        m_two: dict, default is None
            The metric filter to use to assess portfolio performance
        executed: bool, default is False
            If the current portfolio test described by current_state has already been executed


    Returns: list
        The assets filtered which can be traded in the next period
    """

    strat_classes = current_state['strategies']
    if 'cointegration' in current_state['strategies'].keys():
        n = [tuple([x,y]) for x,y in zip(n[::2],n[1::2])]

    n = [list(combinations(n, x)) for x in range(1, len(n) + 1)][2:len(n)-2]
    if 'cointegration' in current_state['strategies'].keys():
        for x in n:
            for i in range(len(x))  :
                x[i] = [str( __ ) for __ in np.ravel(x[i]) ]


    filters = current_state['filters'].copy()
    if 'none' in filters.keys():
        filters.pop('none')
    time_period = current_state['time_period']
    name = current_state['name']

    if executed: return [tuple(x) for x in
                         pd.read_parquet(current_state['name']).sort_values(by=str(time_period) + ' Alpha').index][-1:]

    if not m_two: m_two = dict(Total_Return=False, Sharpe=False, Alpha=False, Number_of_Trades=False)

    runner_multiple([tuple(y) for x in n for y in x], [time_period],
                    _port_sim, init_money=1000, strat_class=strat_classes,
                    inputs=None, num_processes=num_processes,
                    output_metrics=m_two,
                    freq='d', weights_filter=filters, graphs=False, parallel=parallel
                    ).to_parquet(
        name)
    if pd.read_parquet(name).empty:
        return []
    k = pd.read_parquet(name).sort_values(by=str(time_period) + ' Sharpe').index[-1:]
    if optimize:
        strat_class = dict.fromkeys(strat_classes.keys())
        for x, y in strat_classes.items():
            strat = dict.fromkeys(y.keys())
            for k, z in y.items():
                if k == 'roll':
                    strat[k] = (10,50,5)
                elif k == 'z_threshold':
                    if type(z) == tuple:
                        strat[k] = tuple((1.1, 2, .2) for x in range(len(z)))
                    else:
                        strat[k] = (1.1,2,.2)
            strat_class[x] = strat

        params,value = parameter_optimization(type=type_,stck_list=list(k),init_money=1000, strat_class=strat_class,
                    inputs=None, num_processes= num_processes  ,
                    output_metrics=m_two,
                    freq='d', weights_filter=filters, graphs=False, parallel=parallel
                   ,time_period=time_period)
        print(params)

    return [tuple(x) for x in k]


def testing(k, current_state, visualize=False, executed=False, parallel=False):
    """
    Executes the chosen portfolio over the testing period

    Args:
        k: list
            Assets available for trade in the portfolio
        current_state: dict
        visualize: bool, default is False
        executed: bool, default is False
        parallel: bool, default is False

    Returns: None
    """
    time_period = current_state['time_period']
    strat_classes = current_state['strategies']
    name = current_state['name']
    filters = current_state['filters'].copy()
    if 'none' in filters.keys():
        filters.pop('none')

    if visualize:
        get_analysis(
            k , parameters=[time_period],
            filter_func=_port_sim, init_money=1000, strat_class=strat_classes,
            inputs=None, num_processes=1,
            output_metrics= dict(Total_Return=False, Sharpe=False, Alpha=False, Number_of_Trades=False) ,
            freq='d', weights_filter=filters, graphs=True, )

    if executed: return True

    runner_multiple(k, [time_period],
                    _port_sim, init_money=1000, strat_class=strat_classes,
                    inputs=None, num_processes=1,
                    output_metrics=dict(Total_Return=False, Sharpe=False, Alpha=False, Number_of_Trades=False),
                    freq='d', weights_filter=filters, graphs=False, parallel=parallel
                    ).to_parquet(name)
    results = pd.read_parquet(name)
    if results.empty: return False
    return True

def store_json(current_state):
    """

    Args:
        current_state: dict
    """
    path = Path(__file__).parents[1]

    path = str(path)
    try:
        with open(path + '/artifacts/store_results_v.json', 'r') as f:
            temp = json.load(f)
    except json.decoder.JSONDecodeError:
        temp = []
    cur_state = current_state.copy()
    if current_state['type'] == 'testing':
        obj = pd.read_parquet(current_state['name'])
        cur_state['metrics'] = obj.to_dict(orient='records')[0]

    temp.append(cur_state)
    with open(path + '/artifacts/store_results_v.json', 'w') as f:
        json.dump(temp, f, indent=4)


def checker(current_state, cursor):
    """

    Args:
        current_state: dict
        cursor: DuckDBPyConnection

    Returns: bool

    """
    first_state = _get_relationships(cursor)[0].copy()
    cur_state = dict()
    cur_state['time_period'] = str(list(current_state['time_period']))
    cur_state['type'] = current_state['type']
    cur_state['reference_id'] = current_state['reference_id']
    cur_state['reference_id_'] = current_state['reference_id_']
    bol_value = all([None == x for x in first_state.values()])

    if bol_value: return True
    bol = False
    for x in _get_relationships(cursor):

        bol = (x['strategies'] == current_state['strategies']) & (cur_state.items() <= x.items()) & (
                    set(x['filters']) == set(current_state['filters'].keys()))

        if bol:
            current_state['name'], current_state['reference_id'], current_state['reference_id_'] = x['name'], x[
                'reference_id'], x['reference_id_']
            break

    not_executed = not bol

    return not_executed


def updater(not_executed, current_state, cursor, path):
    """

    Args:
        not_executed: bool
        current_state: dict
        cursor: DuckDBPyConnection
        path: str

    Returns: str,int

    """
    if not_executed:
        _database_update(cursor, current_state, )
        store_json(current_state)

    id_ = cursor.execute('SELECT id FROM results  WHERE name = ? ', [current_state['name']]).fetchall()[0][0]
    results_id = _col_length_database(cursor, 'results')

    name = path + '/artifacts/files/' + str(results_id) + '.parquet'

    return name, id_


def run(periods=4, portfolio=None, period_lengths=(500, 260, 150), roll=500):
    """
    Executes the backtesting process.

    See example() below for further details.

    Args:
        periods: int, default is 4
            number of periods to run.
        portfolio: tuple, default is None
            Defines a portfolio
        period_lengths: tuple, default is (500,260,150)
             Define the length of the training, validation, and testing periods in
             that order
        roll: int, default is 500
            Size of Rolling window
    Returns: None
    """
    path = Path(__file__).parents[1]

    path = str(path)
    cursor = _database_create(path    )

    for period_count in range(0, periods):

        u = pd.read_parquet(path + '/data/processed/close_1d_10y.parquet').columns

        results_id = _col_length_database(cursor, 'results')
        name = path + '/artifacts/files/' + str(results_id) + '.parquet'
        time_period = (roll * period_count, period_lengths[0] + roll * period_count)

        current_state = portfolio[0].copy()

        if 'initial_filter_period' in current_state.keys():
            initial_filter_period = current_state.pop('initial_filter_period')
        else:
            initial_filter_period = None
        if 'metrics' in current_state.keys():
            metrics = current_state.pop('metrics')
        else:
            metrics = None
        if 'n_count' in current_state.keys():
            n_count = current_state.pop('n_count')
        else:
            n_count = 10
        if 'parallel' in current_state.keys():
            parallel = current_state.pop('parallel')
        else:
            parallel = True
        if 'visualize' in current_state.keys():
            visualize = current_state.pop('visualize')
        else:
            visualize = False

        current_state['name'] = name
        current_state['type'] = 'training'
        current_state['time_period'] = time_period
        current_state['reference_id'] = -1
        current_state['reference_id_'] = -1

        not_executed = checker(current_state, cursor)

        n = training(current_state, list (u), metrics=metrics, n_count=n_count, initial_filter_period=initial_filter_period,
                     executed=not not_executed, parallel=parallel ,visualize=visualize )
        print(pd.read_parquet(current_state['name']))
        print(not_executed)

        n = [x[0] for x in n]
        name, training_id = updater(not_executed, current_state, cursor, path)



        if not n: continue

        time_period = (time_period[1], time_period[1] + period_lengths[1])

        current_state = portfolio[1].copy()

        if 'metrics' in current_state.keys():
            metrics = current_state.pop('metrics')
        else:
            metrics = None
        if 'parallel' in current_state.keys():
            parallel = current_state.pop('parallel')
        else:
            parallel = True
        if 'optimize' in current_state.keys():
            optimize = current_state.pop('optimize')
        else:
            optimize = False
        current_state['type'] = 'validation'
        current_state['name'] = name
        current_state['time_period'] = time_period

        current_state['reference_id'] = training_id
        current_state['reference_id_'] = -1

        not_executed = checker(current_state, cursor)

        k = validating(n=list (n), current_state=current_state, executed=not not_executed, num_processes= 17, m_two=metrics,
                       optimize = optimize,parallel=parallel)
        print(pd.read_parquet(current_state['name']))
        print(not_executed)

        name, v_id = updater(not_executed, current_state, cursor, path)

        if not k:  continue
        time_period = (time_period[1], time_period[1] + period_lengths[2])
        current_state = portfolio[2].copy()



        if 'visualize' in current_state.keys():
            visualize = current_state.pop('visualize')
        else:
            visualize = False

        current_state['time_period'] = time_period
        current_state['name'] = name

        current_state['type'] = 'testing'

        current_state['reference_id'] = training_id
        current_state['reference_id_'] = v_id

        not_executed = checker(current_state, cursor)

        results = testing(list(k), current_state, executed=not not_executed,visualize=visualize  )

        print(pd.read_parquet(current_state['name']  ))
        print(not_executed)
        if not results: continue
        updater(not_executed, current_state, cursor, path)


