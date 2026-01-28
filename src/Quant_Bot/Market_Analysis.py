import calendar,datetime,io,json,sys
import os

import numpy as np
import pandas as pd
import yfinance
import yfinance as yf

from alpaca.data.timeframe import TimeFrameUnit,TimeFrame
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest,StockLatestTradeRequest,StockLatestQuoteRequest
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


pd.set_option('display.max_row', None)
pd.set_option('display.max_column', None)
pd.set_option('display.width', 0)
np.set_printoptions(threshold=np.inf)


API_KEY = ''
SECRET_KEY = ''
BASE_URL = 'https://paper-api.alpaca.markets'

client = StockHistoricalDataClient(api_key=API_KEY,secret_key=SECRET_KEY)

# List of stock pairs needed to get the data
## This should be either customized in terms of which stocks to get the data of
full_stocks = os.listdir('Extras/Stocks')

#shift
def get_time_period( args, custom_data = False, num_data_points = 100,freq='d',details=False,shift =1):

    if custom_data:
        if freq == 'd':
            data = pd.read_parquet('Close10y1d.parquet').iloc[shift:num_data_points + shift][args].dropna()
        elif freq == '5m':

            data = pd.read_parquet('Closemax5m.parquet').iloc[shift:num_data_points + shift]
            data = data[args].dropna()
        elif freq == '15m':
            data = pd.read_parquet('Closemax15m.parquet').iloc[shift:num_data_points + shift][args].dropna()
        elif freq=='h':
            data = pd.read_parquet('Closemax1h.parquet').iloc[shift:num_data_points + shift][args].dropna()

        else:
            return None
    else:
        if freq == 'd':
            tm = datetime.datetime.now() - datetime.timedelta(days =num_data_points)*5
            timeframe = TimeFrame.Day
        elif freq == 'h':
            timeframe = TimeFrame.Hour
            tm = datetime.datetime.now() - datetime.timedelta(hours =num_data_points)*10

        elif freq == '5m':
            timeframe = TimeFrame(5,TimeFrameUnit.Minute)
            tm = datetime.datetime.now() - datetime.timedelta(minutes =5 * num_data_points)*1000

        elif freq == '15m':
            timeframe = TimeFrame(15,TimeFrameUnit.Minute)
            tm = datetime.datetime.now() - datetime.timedelta(minutes = 15 * num_data_points)*100
        elif freq == '30m':
            timeframe = TimeFrame(15 * 2, TimeFrameUnit.Minute)
            tm = datetime.datetime.now() - datetime.timedelta(minutes=15 *( 2 + 1 -1 ) * num_data_points)*1000
        else:
            return None
        data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

        stock_bars = data_client.get_stock_bars(
            request_params=StockBarsRequest(timeframe=timeframe, symbol_or_symbols=args, start=tm)).df['close'].reset_index()
        data = pd.DataFrame()
        for x in args:
            close = stock_bars[stock_bars['symbol'] == x]
            if data.empty:
                data[x] = list(close['close'])
                data.index = list(close['timestamp'])
            else:
                data[x] = pd.Series(index=list(close['timestamp']),data=list(close['close']))
    return data

def get_yf(per,intr,stocks=tuple(full_stocks)) -> None:
    t = yfinance.download(period=per,interval= intr,tickers=stocks + ['SPY'])['Close']
    results = (t.isna().sum() > int(1/1000*len(t.index)))
    t[[x for x in results.index if not results[x]]].to_parquet('Close' + per + intr + '.parquet')
    return

