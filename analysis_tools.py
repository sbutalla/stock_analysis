import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.mc as mc
import yfinance as yf


matplotlib.rcParams.update({'font.size': 28}) # increase fontsize for matplotlib

def format_path(path, verbose = False):
    '''
    Formats a path by removing trailing '/'
    and creates directory if it does not existsto avoid errors
    Arguments:
        Positional:
            path:   (str) 
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
    plot_close:    Creates a line plot and (optionally) overlays simple
                   moving average curves.
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
            nyse_err   = "  --> NYSE tickers can have a max of 4 letters"
            nasdaq_err = "  --> Nasdaq tickers can have a max of 5 letters"

            raise ValueError("Error: Stock ticker symbol length requirements:\n%s\n%s" % (nyse_err, nasdaq_err))

            
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


    def plot_close(self, sma = False, window = None, save_fig = False, path = None):
        '''
        Plots a candlestick chart and the associated volume plot.
        Arguments:
            Positional: None
              Default:
                  sma:       (bool) Adds a plot of the SMA(s) provided in the 'window'
                             default argument.
                  window:    (int, list) The window size(s) for the SMA(s). Can be passed
                             as either an int or a list of ints.
                  save_fig:  (bool) Saves the plot.
                  path:      (str) The absolute or relative path to save the
                             file at.       
        '''
        ## check inputs
        if sma:
            if window is None: 
                raise ValueError("Window must be specified as an int or list of ints for SMA calculation")
            elif type(window) not in [int, list]:
                raise TypeError("Window must be types int or list for SMA calculation")aise ValueError("Window must be specified as an int or list of ints for SMA calculation")

        if type(window) is list:
            window      = [int(win_size) for win_size in window]   # make sure all window sizes are ints
            col_palette = cm.hsv(np.linspace(0, 0.6, len(window))) # initialize color palette with the hue, saturation,
                                                                   # value (hsv) colormap
        else:
            window      = [int(window)] # make a size one list
            col_palette = ['crimson']   # initialize default color for a single SMA 

        fig, ax = plt.subplots(figsize = (18, 10))
        ax.grid()
        ax.set_title(self.ticker)
        ax.set_xlabel("Date", loc = 'right')
        ax.set_ylabel("Price (USD)", loc = 'top')
        ax.plot(self.dates, self.close_p, color = "k", label = "Close price") # plot close price as a black line

        if sma:
            for win, color in zip(window, col_palette):     # loop over each window and color in color palette
                sma_vals, sma_dates = self.calc_sma_np(win) # use the numpy method to calculate the SMA
                ax.plot(sma_dates, sma_vals, color = color, label = "SMA (%i day)" % win)

        ax.legend()
        fig.tight_layout()
        plt.show()

        if save_fig:
            if sma:
                plot_name = "%s_close_price_sma_" % self.ticker
                for win, cnt in zip(window, np.arange(len(window))):
                    if cnt < (len(window) - 1):
                        plot_name += "%d_" % win
                    else:
                        plot_name += "%d.pdf" % win
            else:
                plot_name = "%s_close_price.pdf" % self.ticker

            if path is not None:
                path = format_path(path)
            else:
                path = ''

            fig.savefig(path + plot_name)    
