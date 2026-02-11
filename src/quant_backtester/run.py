from backtester_overview import *
import warnings 
warnings.filterwarnings('ignore')

def run():
    # List of stocks which were tested. Obtained with extraction from the Yahoo finance with the condition that there are
    ##no NA values over the last 10 years. Length of data is 2515 points (last 10 years) and there are 46 6 stocks
    stck_list = pd.read_parquet('Close10y1d.parquet').columns

    # Mean reversion is tested on the assumption these stocks returns series all mean revert
    for x in range(0, 2):
        time_period = (500 * x, 500 * (x + 1 ))

        name = '../' +  'MV/results_' + str(time_period) + '.parquet'
        runner_multiple(pd.DataFrame(index=[tuple([x]) for x in stck_list]), [time_period],
                        port_sim, init_money=1000, strat_class='mv',
                        inputs=None, num_processes=16,
                        output_metrics=['Total Return', 'Sharpe', 'Alpha', 'Number of Trades'],
                        freq='d', graphs=False, parameters_=[1.5, 26]).to_parquet(name)

        parameter_optimization(time_period, 'mv')

    # Cointegration 
    # Creates stock pairs which are unique
    stck_list = set(frozenset([x, y]) for x in stck_list for y in stck_list if x != y)
    stck_list = list(stck_list)[:100]
    for x in range(0, 2):
        time_period = (500 * x, 500 * (x + 1 ))
        name = '../' +  'cointegration/results_' + str(time_period) + '.parquet'

        # Filters the given list of stock pairs
        runner_multiple(pd.DataFrame(index=[tuple(x) for x in stck_list]), [time_period],
                        cointegration_filter, init_money=1000, strat_class='cointegration',
                        inputs=None, num_processes=16,  output_metrics=[''],
                        freq='d', graphs=False, parameters_=[1.5, 26]).to_parquet(name)
        #
        parameter_optimization(time_period, 'cointegration')

    # Visualization
    for x in range(0, 2 ):
        time_period = (500 * x, 500 * (x + 1 ))

        name = '../' +  'cointegration/results_' + str(time_period) + '.parquet'
        stck_list = pd.read_parquet(name).index[:]
        time_period = (time_period[0] + 200 + 300, time_period[1] + 255)

        get_analysis(
            pd.DataFrame(index=[tuple(x) for x in stck_list]), parameters=[time_period],
            filter_func=port_sim, init_money=1000, strat_class='cointegration',
            inputs=None, num_processes=16,  output_metrics=[''],
            freq='d', graphs=True, parameters_=[1.5, 26])



if __name__ == '__main__':
    run()
