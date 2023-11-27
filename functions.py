import time
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
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


def gaussian_fit(returns, reph_returns):
     mu = returns.mean()
     sigma = returns.std()

     a = 1/(sigma*np.sqrt(2*np.pi))
     power = (-1*((reph_returns - mu)**2))/(2*(sigma**2))
     return a*np.exp(power)


# Defining a function that constructs the density function of a given distribution using KDE method
def density_function(returns, size, common_norm = True, bw_method='silverman', clf=True, *args, **kwargs):
   """
   Constructs the pdf, f(x), of a given distribution, x (price returns) using Kernel Density Estimation method.
   For additional information visit:
        . https://en.wikipedia.org/wiki/Kernel_density_estimation#References
        . https://seaborn.pydata.org/generated/seaborn.kdeplot.html
   Arguments:
           x: the grid (price returns)
           N: the size of the reconstructed data
           bw_method: bandwidth (smothing factor) of the Kernel Density Estimator. Default is set to 'silverman'.
           """
   graph_data = sns.kdeplot(data=returns,
                            bw_method=bw_method, gridsize=size,
                            *args, **kwargs)
   x = graph_data.get_lines()[0].get_xdata()
   y = graph_data.get_lines()[0].get_ydata()
   if clf:
       plt.clf()
   return pd.DataFrame({"Rephurbished Returns": x, "Probability Density": y})