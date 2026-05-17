
from run import *

def example():
    """
       An example implementation of a backtesting process.
    The setup can be varied depending on user goals and interest, along with varying approaches.
    Note the filters and strategies below can and should be changed to any the user sees fit.

    Assets are chosen by default from SP.
    Filters, strategies, and their respective, if necessary, parameters can be chosen for each period.

    Every state is defined by a time period, a set of tradable assets, a set of exec strategies, a set of applicable filters, and a set of metrics to filter the results

    The training and validation periods are used to test different portfolios built from the given assets, strategies, and filters.

    The testing period is for the execution of the chosen portfolio(s) to evaluate

    portfolio = (training_state, validating_state,testing_state)

        training_state = dict(assets=u, strategies=y_training, filters=f_training, metrics=m_training)

            Optional: initial_filter_period

        validating_state = dict(assets=n, strategies=y_validating, filters=f_validating, metrics=m_validating)

        testing_state = dict(assets=u, strategies=y_testing, filters=f_testing, metrics=m_testing)

    Returns: None
    """

    training_state = dict()

    filters_training = dict()
    # filters_training['none'] = dict()
    filters_training ['regime_estimator'] = dict(roll=20, z_score=2, half_life=20)

    # filters_training['pc'] = dict(n=1, roll=30)
    # filters_training['beta'] = dict(roll=30)
    # filters_training['dollar'] = dict(param=None)
    # filters_training['rebalance'] = dict(roll=30, metrics = ('sharpe','std'))
    training_state['filters'] = filters_training

    metrics_training = dict(Total_Return= .00001  , Sharpe=  1.7  , Alpha=False, Number_of_Trades=False)
    training_state['metrics'] = metrics_training

    training_state['strategies'] = dict()
    training_state['strategies']['momentum_trending'] = dict(z_threshold=  1.99  , roll=35)
    # training_state['strategies']['cointegration'] = dict(z_threshold = 1.7, roll = 25 )
    training_state['strategies']['cross_asset_mv'] = dict(z_threshold= 2.1)
    training_state['strategies']['mv'] = dict(z_threshold=1.9283 , roll=35)
    training_state['strategies']['cross_asset_momentum_trending'] = dict(z_threshold=1.9283 , roll=35)

    training_state['n_count'] = 10

    # training_state['parallel'] = False


    validating_state = dict()

    filters_validating = dict()
    # filters_validating['none']= dict()

    # filters_validating['regime_estimator'] = dict(roll=20, z_score=2, half_life=20)
    # filters_validating['none'] = dict()
    filters_validating['pc'] = dict(n=1, roll=30)
    filters_validating['beta'] = dict(roll=30)
    filters_validating['dollar'] = dict(param=None)
    # filters_validating['rebalance'] = dict(roll=20,metrics=['sector','sharpe'],type   ='neutrality')
    validating_state['filters'] = filters_validating
    # validating_state['optimize'] = True

    validating_state['metrics'] =  dict(Total_Return= False, Sharpe=   1 , Alpha= False, Number_of_Trades=False)
    validating_state['strategies'] = dict()

    validating_state['strategies']['momentum_trending'] = dict(z_threshold= 1.99987  , roll=30)
    # validating_state ['strategies']['mv'] = dict(z_threshold= 1.998897 , roll=30)
    # validating_state['strategies']['cross_asset_mv'] = dict(z_threshold=  2 )
    # validating_state ['strategies']['cross_asset_momentum_trending'] = dict(z_threshold=1.989)
    # validating_state ['strategies']['cointegration'] = dict(z_threshold = 1.7, roll = 25 )
    validating_state['strategies']['mv'] = dict(z_threshold= 1.99987  , roll=30)

    # validating_state['parallel'] = False


    testing_state = dict()

    filters_testing = dict()
    # filters_testing['none'] = dict   (   )
    # filters_testing['none'] = dict()
    # filters_testing['regime_estimator'] = dict(roll=20, z_score=2, half_life=20)
    # filters_testing['pc'] = dict(n=1, roll=30)
    # filters_testing['beta'] = dict(roll=30)
    # filters_testing['dollar'] = dict(param=None)
    # filters_testing['rebalance'] = dict(roll=20,metrics=['sharpe'])
    testing_state['filters'] = filters_testing

    testing_state['strategies'] = dict()
    # testing_state['strategies'] ['mv'] = dict(z_threshold=1.99,roll=30)
    testing_state['strategies']['cross_asset_mv'] = dict(z_threshold=1.999 )
    testing_state['strategies']['momentum_trending'] = dict(z_threshold= 1.9999998 , roll= 30)
    # testing_state  ['strategies']['cointegration'] = dict(z_threshold = 1.7, roll = 25 )

    # testing_state  ['strategies']['cross_asset_momentum_trending'] = dict(z_threshold=1.999)
    testing_state['visualize'] = True

    testing_periods = (500,260,260)
    portfolio = (training_state, validating_state, testing_state)
    run(portfolio=portfolio,period_lengths=testing_periods)


if __name__ == '__main__':
    example()
