import warnings

from quant_backtester import *
from quant_backtester.market_filters_analysis import get_analysis
import pandas as pd, optuna, numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', lambda x: str(round(float(x), 4)))
pd.set_option('display.precision', 4)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

np.set_printoptions(suppress=True)
optuna.logging.set_verbosity(optuna.logging.ERROR)


def run():
    """
    Executes the backtesting process.
    Total number of windows can be chosen.
    time_period: Determines the time period of one window
        training_period: Roughly 500 points
        validating_period/ testing_period: Optional validation and separate testing period, or the validation period
        can be treated as the testing period itself.
    strat_classes: List of possible strategy classes which have been implemented.
        'cointegration', 'mv' (mean reversion ), 'cross_asset_mv', 'momentum_trending'
    Results are stored in parquet files, as dataframes, with indices being the stocks in that portfolio, and the columns
    defining possible metrics of evaluating the performance.

    Below is given one example of a backtesting process. The setup can be varied depending on user goals and interest, along with varying approaches.
    Note the filters below can and should be changed to any the user sees fit.

    In this example, a simple regime filter is used to test if the market is in a low or high trend latent state. Mean reversion or momentum trending is applied
    depending on the result as a filter to selected stocks. These stocks are chosen themselves from a correlation filter (and possibly a market cap filtering method)
    which acts on the first half of the training time period. These stocks are then trained on the later half of the
    training period, and are treated as individual portfolios (singleton asset universes) to determine if they are profitable (defined as a positive total return and
    a Sharpe ratio higher than a cutoff of 1.777). Then, the resulting stocks are filtered in terms of the three with the highest relative individual alphas on their individual portfolios,
    and these are tested in various combinations on the validation period. The best performing combination (highest alpha) is then executed on the testing period as a simulation of a live trading.
    A roll forward of 500 occurs, and this whole process is repeated.

    The setup for cointegration is given also as comments below.

    Note all the possible types, and classes of filters are defined in the market_filters module.

    :return: None
    """
    # Storage of current path for ease in creation of cache files.
    path = Path(__file__).parents[1]
    path = str(path)
    #
    # Run the following with a list of stocks (stck_list) to get a filtered dataframe of closing prices over the specified time period.
    # Stores locally as a cached file (parquet, not committed)
    ##    get_yf('10y','1d',stck_list)

    # Regime detector is not robust currently, meaning it may be better to define what strategy to apply manually for testing purposes.
    strat_classes = 'cross_asset_mv',
    for strat_class in strat_classes:
        Path.mkdir(Path(str(path) + '/artifacts/' + strat_class), parents=True, exist_ok=True)
        for x in range(0, 4):
            # Training

            # Universe of possible stocks
            stck_list = pd.read_parquet(path + '/data/processed/close_1d_10y.parquet').columns

            time_period = (500 * x, 500 * (x + 1))

            # Filter period to reduce the size of this universe
            filter_period = (time_period[0], time_period[0] + 250)
            # strat_class=  regime_estimator(dict(stock_list=stck_list, time_period=filter_period, freq='d'))

            # stck_list = market_cap_filter(dict(stock_list=stck_list, time_period=filter_period, freq='d'), 'medium')
            stck_list = corr_filter(dict(stock_list=stck_list, time_period=filter_period, freq='d'), .9)
            print(len(stck_list))
            # Co-integration set up of stock pairs

            # stck_list = set(frozenset([x, y]) for x in stck_list for y in stck_list if x != y and 'SPY' not in [x,y])
            # # stck_list = list(stck_list)[:1000]
            # stck_list = runner_multiple(pd.DataFrame(index=[tuple(x) for x in stck_list  ]), [filter_period],
            #                 cointegration_filter, freq = 'd'  ,     strat_class=strat_class,
            #                 inputs=None, graphs = False, output_metrics = dict(p=.05), num_processes=16,
            #
            #                             )
            # stck_list = stck_list.index
            # Simulated backtest to determine individual asset behavior, which acts as
            testing_period = (filter_period[1], time_period[1])
            z_threshold, roll = 1.5, 14
            name = str(path) + '/artifacts/' + strat_class + '/results_training_' + '(' + str(z_threshold) + ',' + str(
                roll) + ')_' + str(testing_period) + '.parquet'
            # runner_multiple(pd.DataFrame(index=[tuple([x]) for x in list(tuple(stck_list))]  ), [testing_period],
            #                 _port_sim, init_money=1000, strat_class=strat_class,
            #                 inputs=None, num_processes=16,
            #                 output_metrics=dict(Total_Return= 0 , Sharpe= 1.777     , Alpha=False, Number_of_Trades=False),
            #                 freq='d',weights_filter = dict(   ) ,     graphs=False, parameters_=dict(z_threshold=z_threshold, roll=roll)).to_parquet(
            #     name)
            # print(pd.read_parquet(name).sort_values(by=str(testing_period) + ' Alpha'))

            # continue

            stck_list = pd.read_parquet(name).sort_values(by=str(testing_period) + ' Alpha')
            stck_list = stck_list.where(stck_list[str(testing_period) + ' Alpha'] > .8, pd.NA).dropna()
            if stck_list is None: continue
            stck_list = [x[0] for x in list(stck_list.index[-2-1:])]

            time_period = (time_period[1], time_period[1] + 255)
            str_ = '(' + str(z_threshold) + ',' + str(roll) + ')_to' + '_'
            z_threshold, roll = z_threshold + .2, roll
            name = str(path) + '/artifacts/' + strat_class + '/results_validating_' + str_ + '(' + str(
                z_threshold) + ',' + str(
                roll) + ')_' + str(time_period) + '.parquet'
            # parameter_optimization(dict( strat_class=strat_class, stock_list = stck_list,        freq='d'    ,time_period=time_period), 'grid' )
            runner_multiple(pd.DataFrame(index=[tuple(stck_list )    ]), [time_period  ],
                            _port_sim, init_money=1000, strat_class=strat_class,
                            inputs=None, num_processes=16,
                            output_metrics=dict(Total_Return= False , Sharpe=False         , Alpha=False, Number_of_Trades=False),
                            freq='d',weights_filter = None  ,     graphs=False, parameters_=dict(z_threshold=z_threshold, roll=roll)).to_parquet(
                name)

            stck_list = pd.read_parquet(name).sort_values(by=str(time_period) + ' Alpha').index[:10]
            print(pd.read_parquet(name).sort_values(by=str(time_period) + ' Alpha'))
            stck_list = list(stck_list[0])
            if stck_list is None: continue

            # continue

            time_period = (time_period[1], time_period[1] + 150)

            name = str(path) + '/artifacts/' + strat_class + '/results_testing_' + str(time_period) + '.parquet'
            runner_multiple(pd.DataFrame(index=[tuple(stck_list)]), [time_period],
                            _port_sim, init_money=1000, strat_class=strat_class,
                            inputs=None, num_processes= 1 ,
                            output_metrics=dict(Total_Return=False, Sharpe=False, Alpha=False, Number_of_Trades=False),
                            freq='d',weights_filter = None ,  graphs=False, parameters_=dict(z_threshold=1.5, roll=26)).to_parquet(name)
            print(pd.read_parquet(name).sort_values(by=str(time_period) + ' Alpha'))

            # Visualization
            get_analysis(
                pd.DataFrame(index=[tuple(stck_list)]), parameters=[time_period],
                filter_func=_port_sim,init_money=1000, strat_class=strat_class,
                            inputs=None, num_processes= 1 ,
                            output_metrics=dict(   ),
                            freq='d',weights_filter = None ,  graphs=True, parameters_=dict(z_threshold=1.5, roll=26))


        break


if __name__ == '__main__':
    run()
