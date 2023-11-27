import time
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# Defining a class that stores historical data
class yahoodata:
    
    """
    Imports histroical data from yahoo finance for the given parameters:

    ticker: symbol
    period1&2: starting and ending dates
    interval: time-scale ("1d": daily, "1mo":monthly)
    
    The getdata() method gives the entire historical data (OHLCV)
    """
    
    def __init__(self, ticker, period1, period2, interval):
        self.ticker = ticker
        self.period1 = period1
        self.period2 = period2
        self.interval = interval


    def getdata(self):
        history = yf.download(tickers=self.ticker,
                              start=self.period1,
                              end=self.period2,
                              interval=self.interval)
        return history.dropna().reset_index().set_index("Date")
    

# Defining a function that calculates log-returns of a given security
def log_returns(historical_data):
        """
        Historical data must be a numpy array or pandas dataframe.
        """
        returns = np.log(historical_data) - np.log(historical_data.shift(1))
        return returns