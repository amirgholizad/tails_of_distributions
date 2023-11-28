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
        Historical data must be a numpy array or pandas dataframe
        """
        returns = np.log(historical_data) - np.log(historical_data.shift(1))
        return returns


def gaussian_fit(returns, reph_returns):
    """
    Calculates the gaussian function for a given:
        reph_returns: preferably a uniform grid
        returns: original return values for obtaining mean and standard deviation
    """
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


# creating a function that finds the limits of tails
def intersection_points(Return, PDF=[], GDF=[], grid_size=1000):
    """
    Finds the indices of where the reph_returns == R1 & R2:
        Return: a dataframe containing rates of return
        PDF: generated probability density function using KDE method
             if not given, will be calculated
        GDF: gaussian function from reph_return
             if not given, will be calculated
    """
    first_quantile = Return.mean() - Return.std()
    third_quantile = Return.mean() + Return.std()

    if len(PDF) == 0 and len(GDF) == 0:
        PDF = density_function(returns=Return, size=grid_size, common_norm=False, clf=True)
        GDF = gaussian_fit(returns=Return, reph_returns=PDF["Rephurbished Returns"])

    first_intersection = abs(PDF["Rephurbished Returns"] - first_quantile).argmin()
    second_intersection = abs(PDF["Rephurbished Returns"] - third_quantile).argmin()

    return [first_intersection, second_intersection]

# a function that returns the area under the tails
def tails(Return, PDF=[], GDF=[], grid_size=1000):
     
     """
    Calculates the difference between the tails of gaussian and KDE fits:
        Return: a dataframe containing rates of return
        PDF: generated probability density function using KDE method
             if not given, will be calculated
        GDF: gaussian function from reph_return
             if not given, will be calculated
    """
     if len(PDF) == 0 and len(GDF) == 0:
        PDF = density_function(returns=Return, size=grid_size, common_norm=False, clf=True)
        GDF = gaussian_fit(returns=Return, reph_returns=PDF["Rephurbished Returns"])

     inter_points  = intersection_points(Return, PDF, GDF, grid_size)

     gdf_left = np.trapz(y = GDF[:inter_points[0]], x = PDF["Rephurbished Returns"][:inter_points[0]])
     gdf_right = np.trapz(y = GDF[inter_points[1]:], x = PDF["Rephurbished Returns"][inter_points[1]:])
     gdf_total = gdf_left + gdf_right

     pdf_left = np.trapz(y = PDF["Probability Density"][:inter_points[0]], x = PDF["Rephurbished Returns"][:inter_points[0]])
     pdf_right = np.trapz(y = PDF["Probability Density"][inter_points[1]:], x = PDF["Rephurbished Returns"][inter_points[1]:])
     pdf_total = pdf_left + pdf_right
    
     return pdf_total - gdf_total


# plotting tails
def tails_graph(ticker, returns, interval, grid_size=1000, c2='green'):
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), sharex=True)

    ax1.set_title("Tails of Gaussian vs KDE Distributions ({})".format(ticker), fontweight="bold")
    PDF = density_function(returns[interval], size=grid_size,
                                common_norm=False, color='black',
                                clf=False, ax=ax1, linestyle='--',
                                label=f'KDE fit ({interval})')
    GDF = gaussian_fit(returns=returns[interval], reph_returns=PDF["Rephurbished Returns"])
    inter_points = intersection_points(returns[interval], PDF=PDF, GDF=GDF)
    sns.lineplot(x=PDF["Rephurbished Returns"], y=GDF, ax=ax1, color=c2, label=f'Gaussian fit ({interval})', linestyle='-')
    ax1.set_ylabel(ylabel="$Density(f(q))$", fontweight="bold", style="italic", fontsize = 16)
    ax1.legend()
    ax1.set_xlabel(xlabel="$Return(q)$", fontweight="bold", style="italic", fontsize = 16)
    sns.set(style="whitegrid")


    ax1.fill_between(PDF["Rephurbished Returns"], PDF["Probability Density"], 
                    where = (PDF["Rephurbished Returns"].index < inter_points[0]),
                    interpolate=True, color="grey")
    ax1.fill_between(PDF["Rephurbished Returns"], PDF["Probability Density"], 
                    where = (PDF["Rephurbished Returns"].index > inter_points[1]),
                    interpolate=True, color="grey")
    ax1.fill_between(PDF["Rephurbished Returns"], GDF, 
                    where = (PDF["Rephurbished Returns"].index < inter_points[0]),
                    interpolate=True, color=c2)
    ax1.fill_between(PDF["Rephurbished Returns"], GDF, 
                    where = (PDF["Rephurbished Returns"].index > inter_points[1]),
                    interpolate=True, color=c2)

    ax1.scatter(x=PDF["Rephurbished Returns"][inter_points[0]], y=GDF[inter_points[0]], color='black', marker='o', s=50)
    ax1.scatter(x=PDF["Rephurbished Returns"][inter_points[1]], y=GDF[inter_points[1]], color='black', marker='o', s=50)
    ax1.set_ylim([-0.1, 1.2*PDF["Probability Density"].max()])
    ax1.set_xlim([returns[interval].mean() - 5*returns[interval].std(), returns[interval].mean() + 5*returns[interval].std()])
    plt.savefig(f"results/tails_{interval}.jpg")
    plt.show()
