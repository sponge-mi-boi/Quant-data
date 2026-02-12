import warnings
from pathlib import Path

from data_filter import *
from backtester_overview import *

warnings.filterwarnings('ignore')


def run():
    # List of stocks to be tested. d
    stck_list = pd.read_parquet('Close10y1d.parquet').columns

    # Run the following with stck_list to get a filtered dataframe of closing prices over the specified time period.
    # Stores locally as a cached file (parquet, not committed)
    #    get_yf('10y','1d',stck_list)

    path = Path(__file__).parents[2]
    path = str(path)

    # Mean reversion is tested on the assumption these stocks returns series all mean revert
    # for x in range(0, 2):
    #     time_period = (500 * x, 500 * (x + 1))
    #
    #     name =  str ( path ) + '/artifacts/mv/'  + 'results_' + str(time_period) + '.parquet'
    #
    #     runner_multiple(pd.DataFrame(index=[tuple([x]) for x in stck_list]), [time_period],
    #                     port_sim, init_money=1000, strat_class='mv',
    #                     inputs=None, num_processes=16,
    #                     output_metrics=['Total Return', 'Sharpe', 'Alpha', 'Number of Trades'],
    #                     freq='d', graphs=False, parameters_=[1.5, 26]).to_parquet(name)
    #
        # parameter_optimization(time_period, 'mv')

    # Cointegration 
    # Creates stock pairs which are unique
    stck_list = set(frozenset([x, y]) for x in stck_list for y in stck_list if x != y)
    stck_list = list(stck_list)[:100]
    for x in range(0, 2):
        time_period = (500 * x, 500 * (x + 1))

        name = str(path) + '/artifacts/cointegration/results_' + str(time_period) + '.parquet'

        # Filters the given list of stock pairs
        runner_multiple(pd.DataFrame(index=[tuple(x) for x in stck_list]), [time_period],
                        cointegration_filter, init_money=1000, strat_class='cointegration',
                        inputs=None, num_processes=16, output_metrics=[''],
                        freq='d', graphs=False, parameters_=[1.5, 26]).to_parquet(name)
        #
        parameter_optimization(time_period, 'cointegration')

    # Visualization
    for x in range(0, 2):
        time_period = (500 * x, 500 * (x + 1))

        name = str(path) + '/artifacts/cointegration/results_' + str(time_period) + '.parquet'

        stck_list = pd.read_parquet(name).index[:]
        time_period = (time_period[0] + 200 + 300, time_period[1] + 255)

        get_analysis(
            pd.DataFrame(index=[tuple(x) for x in stck_list]), parameters=[time_period],
            filter_func=port_sim, init_money=1000, strat_class='cointegration',
            inputs=None, num_processes=16, output_metrics=[''],
            freq='d', graphs=True, parameters_=[1.5, 26])


if __name__ == '__main__':
    run()
