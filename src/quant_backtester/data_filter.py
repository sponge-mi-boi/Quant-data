import datetime
from pathlib import Path

import pandas as pd
import yfinance
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrameUnit, TimeFrame


class User:
    # These are the custom keys for the user's Alpaca account.
    API_KEY = ''
    SECRET_KEY = ''

    def __init__(self, api_key, secret_key) -> None:
        self.API_KEY = api_key
        self.SECRET_KEY = secret_key

    def get_time_period(self, args, freq='1d', num_data_points=100) -> pd.DataFrame:
        if freq == 'd' or freq == '1d':
            tm_start = datetime.datetime.now() - datetime.timedelta(days=num_data_points) * 5
            timeframe = TimeFrame.Day
        elif freq == 'h':
            timeframe = TimeFrame.Hour
            tm_start = datetime.datetime.now() - datetime.timedelta(hours=num_data_points) * 10

        elif freq == '5m':
            timeframe = TimeFrame(5, TimeFrameUnit.Minute)
            tm_start = datetime.datetime.now() - datetime.timedelta(minutes=5 * num_data_points) * 1000

        elif freq == '15m':
            timeframe = TimeFrame(15, TimeFrameUnit.Minute)
            tm_start = datetime.datetime.now() - datetime.timedelta(minutes=15 * num_data_points) * 100
        elif freq == '30m':
            timeframe = TimeFrame(15 * 2, TimeFrameUnit.Minute)
            tm_start = datetime.datetime.now() - datetime.timedelta(minutes=15 * (2 + 1 - 1) * num_data_points) * 1000
        else:
            return pd.DataFrame()
        data_client = StockHistoricalDataClient(self.API_KEY, self.SECRET_KEY)

        stock_bars = data_client.get_stock_bars(
            request_params=StockBarsRequest(timeframe=timeframe, symbol_or_symbols=args, start=tm_start)).df[
            'close'].unstack(level='symbol')

        return stock_bars


# BASE_URL = 'https://paper-api.alpaca.markets'


def get_time_period(args, type_='Close', freq='1d',
                    time_peri=None) -> pd.DataFrame:
    path = Path(__file__).parents[2]

    if freq == 'd' or freq == '1d':
        name = str(path) + '/data/processed/' + type_.lower() + '_1' + 'd' + '_' + '10y' + '.parquet'
        data = pd.read_parquet(name).iloc[time_peri[0]: time_peri[-1]][args].dropna()
    elif freq == '1mo':
        name = str(path) + '/data/processed/' + type_.lower() + '_' + freq + '_' + 'max' + '.parquet'
        data = pd.read_parquet(name).iloc[time_peri[0]: time_peri[-1]][args].dropna()
        data = data[args].dropna()
    elif freq == '15m':
        name = str(path) + '/data/processed/' + type_.lower() + '_' + freq + '_' + 'max' + '.parquet'
        data = pd.read_parquet(name).iloc[time_peri[0]: time_peri[-1]][args].dropna()
    elif freq == 'h':
        name = str(path) + '/data/processed/' + type_.lower() + freq + '_' + 'max' + '.parquet'
        data = pd.read_parquet(name).iloc[time_peri[0]: time_peri[-1]][args].dropna()
    else:
        return pd.DataFrame()

    return data


def get_yf(per, int_, stocks=tuple(), type_='Close') -> None:
    """
    A wrapper on yahoo finance public data for easy creation and storage of new data.
    Note:
        - Applied a selection filter of having less than 1/1000 * (size of the data set) NA values
    """
    path = Path(__file__).parents[2]
    name = str(path) + '/data/processed/' + type_.lower() + '_' + int_ + '_' + per + '.parquet'

    if type_ in ['Close','volume','high','low']:
        data = yfinance.download(period=per, interval=int_, tickers=list(stocks) + ['SPY'])[type_]

    t = data
    results = (t.isna().sum() > int(1 / 1000 * len(t.index)))

    t[[x for x in results.index if not results[x]]].to_parquet(
        name)


    breakpoint()



def get_stock_universe(type_='Close', asset_exchange_or_type='SP') -> list:
    path = Path(__file__).parents[2]

    name = str(path) + '/data/processed/' + type_.lower() + '_1' + 'd' + '_' + '10y' + '.parquet'

    return list(pd.read_parquet(name).columns)

def get_info(type_):
    path = Path(__file__).parents[2]

    aspects = pd.concat ([pd.read_parquet(str(path) + '/data/processed/assets_info_more_detail_05_14_26.parquet'),pd.read_parquet(str(path) + '/data/processed/assets_info_05_14_26.parquet')],axis=1).drop('SPY')


    return aspects[type_]
