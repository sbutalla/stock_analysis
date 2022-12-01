import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import matplotlib
import yfinance as yf


matplotlib.rcParams.update({'font.size': 28})



def format_path(path, verbose = False):
    '''
    Formats a path by removing trailing '/'
    and creates directory if it does not existsto avoid errors
    '''
    if path == '':
        pass
    else:
        if path[-1] != '/':
             path += '/'
        else:
            pass
        
        try: # try making dir if it doesn't already exist
            os.makedirs(path)
            if verbose:
                print('New directory created at %s' % path)
        except FileExistsError:  # skip if dir exists
            pass
    return path

class stock_data:
    '''
    Class for retrieving, processing, and analyzing stock data.
    Methods:
    __init__:      Constructor for the stock_data class. Retrieves the stock data
                   given the input of a ticker symbol, converts the data for the
                   date/time, low, high, opening, closing, and volume to np.arrays
                   and stores as attributes of the class.
    print_summary: Uses pretty print to print the stock information.
    print_news:    Uses pretty print to print the information on the news articles
                   that feature the stock.    
    plot_candle:   Plots a candlestick chart and the associated volume plot.
    '''
    
    def __init__(self, ticker):
        '''
        Constructor for the stock_data class. Retrieves stock data and initializes 
        several variables that are used for analysis/plotting.

        Arguments:
            Positional:
                ticker: (str) Ticker symbol.
        '''
        ## check input
        if len(ticker) > 5:
            print("Error: Stock ticker symbol cannot be more than 5 characters in length!")
            print("  --> NYSE tickers can have a max of 4 letters")
            print("  --> NYSE tickers can have a max of 5 letters")
            
        self.ticker = ticker
        
        self.data    = yf.Ticker(self.ticker)             # store object created from yf.Ticker() method as attr
        self.history = self.data.history(period = "max")  # store pd.DataFrame of max. stock data history as attr
        self.low     = self.history["Low"].to_numpy()     # store np.array of lowest daily stock price as attr
        self.high    = self.history["High"].to_numpy()    # store np.array of highest daily stock price as attr
        self.open_p  = self.history["Open"].to_numpy()    # store np.array of daily opening stock price as attr
        self.close_p = self.history["Close"].to_numpy()   # store np.array of daily closing stock price as attr
        self.volume  = self.history["Volume"].to_numpy()  # store np.array of daily stock trading volume as attr
        self.dates   = self.history.index.to_pydatetime() # store dataframe indices (dates) in pydatetime format as attr
        
    # define the activation function
    def print_summary(self):
        '''
        Uses pretty print to print the stock information.

        Arguments: None
        '''
        pprint(self.data.info)
    
    def print_news(self):
        '''
        Uses pretty print to print the information on the
        news articles that feature the stock.

        Arguments: None
        '''
        pprint(self.data.news)
        
        
    def plot_candle(self, plot_volume = True, save_fig = False, path = None):
        '''
        Plots a candlestick chart and the associated volume plot.

        Arguments:
            Positional: None
            Default:
                plot_volume: (bool) Adds a plot of the trading volume beneath
                             the candlestick chart. Default = True.
                save_fig:    (bool) Saves the plot.
                path:        (str) The absolute or relative path to save the
                             file at.
        '''
        if plot_volume: # add volume plot to the candlestick chart
            y_labels = {0: "Price (USD)", 1: "Number of trades"} # y labels

            fig, ax = plt.subplots(2, 1, figsize = (20, 16)) # create grid of two rows and one column
            ax.ravel() # ravel/flatten axes object for easy indexing
            ax[0].set_title(self.ticker) # set title for candlestick chart only
            for subplot in range(len(ax)): # set grid, x, and y labels for both plots
                ax[subplot].grid()
                ax[subplot].set_xlabel("Date", loc = 'right')
                ax[subplot].set_ylabel(y_labels[subplot], loc = 'top')            
        else: # plot only the candlestick chart
            fig, ax = plt.subplots(figsize = (20, 16))
            ax.grid()
            ax.set_title(self.ticker)
            ax.set_xlabel("Date", loc = 'right')
            ax.set_ylabel("Price (USD)", loc = 'top')

        for point in np.arange(self.high.shape[0]): # loop over each observation
            if self.close_p[point] > self.open_p[point]: # increasing candle
                datum_color = mcolors.CSS4_COLORS["limegreen"]         # candle/volume bar color
                bottom_loc  = self.open_p[point]                       # y location of the bottom of the candle
                body        = self.close_p[point] - self.open_p[point] # length of the candle body
            else: # decreasing candle
                datum_color = mcolors.CSS4_COLORS["crimson"]
                bottom_loc  = self.close_p[point]
                body        = self.open_p[point] - self.close_p[point]

            if plot_volume: 
                ax[0].plot([self.dates[point], self.dates[point]], [self.low[point], self.high[point]], color = "black", solid_capstyle='round', zorder=2) # draw wick
                ax[0].bar(self.dates[point], body, width = 15, color = datum_color, bottom = bottom_loc, edgecolor='black', zorder=3)                      # draw candle body
                ax[1].bar(self.dates[point], self.volume[point], color = datum_color)                                                                      # draw volume bar
            else:
                ax.plot([self.dates[point], self.dates[point]], [self.low[point], self.high[point]], color = "black", solid_capstyle='round', zorder=2) # draw wick
                ax.bar(self.dates[point], body, width = 15, color = datum_color, bottom = bottom_loc, edgecolor='black', zorder=3)                      # draw candle body

        fig.tight_layout()
        plt.show()

        if save_fig:
            if volume:
                plot_name = '%s_candlestick_volume.pdf' % self.ticker
            else:
                plot_name = '%s_candlestick.pdf' % self.ticker
            if path is not None:
                path = format_path(path)
            else:
                path = ''
            fig.savefig(path + plot_name)
    
