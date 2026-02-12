import datetime
from pathlib import Path

import pandas as pd
import yfinance
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrameUnit, TimeFrame

# These are the custom keys for the user's Alpaca account.
API_KEY = ''
SECRET_KEY = ''
#
# BASE_URL = 'https://paper-api.alpaca.markets'

# List of stock pairs to get the data of
full_stocks = []


# Used to dynamically obtain the data of a stock given parameters such as the number of points, which can be based on
# either custom, already obtained data, or dynamically obtained from the alpaca website.

## Live data is obtained only through an approximation of the needed number of data points, meaning it is not exact.
def get_time_period(args, custom_data=False, num_data_points=100, freq='d', time_peri=None) -> pd.DataFrame:
    if custom_data:
        if freq == 'd':
            data = pd.read_parquet('Close10y1d.parquet').iloc[time_peri[0]: time_peri[-1]][args].dropna()
        elif freq == '5m':

            data = pd.read_parquet('Data/Closemax5m.parquet').iloc[time_peri[0]: time_peri[-1]]
            data = data[args].dropna()
        elif freq == '15m':
            data = pd.read_parquet('Data/Closemax15m.parquet').iloc[time_peri[0]: time_peri[-1]][args].dropna()
        elif freq == 'h':
            data = pd.read_parquet('Data/Closemax1h.parquet').iloc[time_peri[0]: time_peri[-1]][args].dropna()

        else:
            return pd.DataFrame()
    else:
        if freq == 'd':
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
        data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

        stock_bars = data_client.get_stock_bars(
            request_params=StockBarsRequest(timeframe=timeframe, symbol_or_symbols=args, start=tm_start)).df[
            'close'].reset_index()
        data = pd.DataFrame()
        for x in args:
            close = stock_bars[stock_bars['symbol'] == x]
            if data.empty:
                data[x] = list(close['close'])
                data.index = list(close['timestamp'])
            else:
                data[x] = pd.Series(index=list(close['timestamp']), data=list(close['close']))
    return data


# A wrapper on yahoo finance public data for easy creation and storage of new data.
def get_yf(per, int_, stocks=tuple(full_stocks), extras='') -> None:
    path = Path(__file__).parents[2]
    name = str(path) + '/data/processed/' + 'close_' + int_  + '_'  + per + '.parquet'

    t = yfinance.download(period=per, interval=int_  , tickers=list(stocks) + ['SPY'])['Close']
    results = (t.isna().sum() > int(1 / 1000 * len(t.index)))
    t[[x for x in results.index if not results[x]]].to_parquet(
       name )
    return
