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

class stockData:
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

    def calc_sma_np(self, window):
        '''
        Calculate the simple moving average.
        Arguments:
          Positional:
            window:   (int; float) The window (of a time period)
                      of the moving average.
        '''
        # array to store sma values        
        sma_vals = np.ndarray((self.close_p.shape[0] + 1 - window,))

        # loop over window
        for ii in range(sma_vals.shape[0]): 
            sma_vals[ii] = np.sum(self.close_p[ii: ii + window]) / window 
    
        sma_dates = self.dates[window - 1:]
        
        return sma_vals, sma_dates

    def calc_ema_np(self, window=None, cm=None, hl=None, theta=None, weighted=False):
        '''
        Calculates the exponential moving average using
        numpy.
        Arguments:
            Keyword:
                window:  (int; float) The window size, w (span).
                cm:      (int; float) The center-of-mass, cm.
                hl:      (int; float) The half-life, t_{1/2}.
                theta:   (int; float) The scaling parameter.
                weighted: (bool) Calculate exponentially weighted moving average.
        '''
        params = [window, cm, hl, theta]
        
        if params.count(None) <= 1 or params.count(None) == 4:
            raise ValueError('Only one variable in [window, cm, hl] can be chosen')
        
        if window:
            sp_val   = window
            var_type = 'win'
        if cm:
            sp_val   = cm
            var_type = 'cm'
        if hl:
            sp_val   = hl
            var_type = 'halflife'
        if theta:
            if theta > 1 or theta < 0:
                raise ValueError('The value of the smoothing parameter (theta) cannot be theta < 0 or 1 < theta')
            else:
                sp_val   = theta
                var_type = 'theta'

        sm_param     = self.smoothing_param(sp_val, var_type) 
        red_sm_param = 1 - sm_param
        num_obs      = self.close_p.shape[0]

        ema_vals    = np.ndarray((num_obs,)) 
        ema_vals[0] = self.close_p[0] # initialize first price of EMA array as p0

        for obs in np.arange(1, num_obs):
            if weighted:
                numer = 0
                denom = 0
                for weight in np.arange(obs): # loop over number of time periods up to the current observation
                    numer += (red_sm_param**weight) * self.close_p[obs - weight]
                    denom += (red_sm_param**weight)

                ema_vals[obs] = numer / denom
            else:
                ema_vals[obs] = (sm_param * self.close_p[obs]) + (red_sm_param * ema_vals[obs - 1])

        return ema_vals

    def calc_boll_np(self, window):
        '''
        Calculates Bollinger Bands using the numpy method.
        Arguments:
            Positional: 
                window:   (int) The window size of the SMA
                          used to calculate the Bollinger
                          Bands.
        '''
        if window not in [10, 20, 50]:
            raise ValueError("Only window sizes of 10, 20, or 50 days currently supported.")
 
        window_to_stddev = {10: 1.5, 20: 2, 50: 2.5}
        band_size        = window_to_stddev[window]
        
        sma_vals, sma_stddev, sma_dates = self.calc_sma_np(window, stddev=True)
        upper_band = sma_vals + (band_size * sma_stddev)
        lower_band = sma_vals - (band_size * sma_stddev)
        
        return sma_vals, upper_band, lower_band, sma_dates
        
        
    def plot_candle(self, plot_volume=True, save_fig=False, path=None):
        '''
        Plots a candlestick chart and the associated volume plot.

        Arguments:
            Keyword:
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


    def plot_close(self, window=None, win_boll=None, span=None, cm=None, hl=None, theta=None, weighted=False,
               verbose=False, save_fig=False, path=None):
        '''
        Plots a candlestick chart and the associated volume plot.
        Arguments:
            Keyword:
                window:    (int; float; list) The window size
                           (lookback period) for the SMA.
                win_boll:  (int) The window size for the Bollinger
                           Band calculation.
                span:      (int; float; list) The span (window size)
                           for the E(W)MA.
                cm:        (int; float; list) The center-or-mass for the
                           E(W)MA.
                hl:        (int; float; list) The half-life for the
                           E(W)MA.
                theta:     (int; float; list) The half-life for the
                           E(W)MA.
                weighted:  (bool) Perform the exponentially weighted moving
                           average (EWMA).
                verbose:   (bool) Increase verbosity.
                save_fig:  (bool) Saves the plot.
                path:      (str) The absolute or relative path to save the
                           file at.       
            '''
        def _gen_col_palette(param_list):
            return matplotlib.cm.hsv(np.linspace(0, 0.6, len(param_list)))      
            
        ## initialize booleans to false for plotting
        sma = False
        ema = False 

        ## check ema arguments
        params = [span, cm, hl, theta] # put params in list for checking
        if params.count(None) < 3:
            raise ValueError('Only one variable in [span, cm, hl, theta] can be chosen')

        ## check inputs
        if window:
            sma = True
            leg_text_sma = '%d-day SMA'
            if type(window) is list:
                col_palette = _gen_col_palette(window)
            else:
                window     = [window]   
                col_palete = ['crimson']

        if span:
            ema = True
            if weighted:
                leg_text_ema = '%d-day EWMA'
            else:
                leg_text_ema = '%d-day EMA'

            if type(span) is list:
                col_palette = _gen_col_palette(span)
            else:
                span       = [span]   
                col_palete = ['crimson']

        if cm:
            ema = True
            if weighted:
                leg_text_ema = r'$C_{m} = %.1f$ EWMA'
            else:
                leg_text_ema = r'$C_{m} = %.1f$ EMA'

            if type(cm) is list:
                col_palette = _gen_col_palette(cm)
            else:
                cm          = [cm]   
                col_palete  = ['crimson']

        if hl:
            ema = True
            if weighted:
                leg_text_ema = r'$t_{1/2} = %.1f$ EWMA'
            else:
                leg_text_ema = r'$t_{1/2} = %.1f$ EMA'

            if type(hl) is list:
                col_palette = _gen_col_palette(hl)
            else:
                hl          = [hl]   
                col_palete  = ['crimson']

        if theta:
            if theta <= 0 or theta > 1:
                raise ValueError('The smoothing parameter (theta) must be 0 < theta <= 1')
                
            ema = True
            if weighted:
                leg_text_ema = r'$\theta = %.2f$ EWMA'
            else:
                leg_text_ema = r'$\theta = %.2f$ EMA'

            if type(cm) is list:
                col_palette = _gen_col_palette(theta)
            else:
                theta       = [theta]   
                col_palete  = ['crimson']
                
        if win_boll:
            color_scheme
            band_color  = color_schemes['green']['band']
            fill_color  = color_schemes['green']['fill']
            band_legend = {10: 1.5, 20: 2, 50: 2.5}


        fig, ax = plt.subplots(figsize=(18, 10))
        ax.grid()
        ax.set_title(self.ticker)
        ax.set_xlabel('Date', loc='right')
        ax.set_ylabel('Price (USD)', loc='top')
        ax.plot(self.dates, self.close_p, color='k', label='Close price')
        
        if sma:
            for win, color in zip(window, col_palette):
                sma_vals, sma_dates = self.calc_sma_np(win)
                ax.plot(sma_dates, sma_vals, color=color, label=(leg_text_sma % win))

        if ema: 
            ema_dates = self.dates            
            if span:
                for val, color in zip(span, col_palette):
                    ema_vals = self.calc_ema_np(window=val, weighted=weighted)
                    ax.plot(ema_dates, ema_vals, color=color, label=(leg_text_ema % val))
            
            if cm:
                for val, color in zip(cm, col_palette):
                    ema_vals = self.calc_ema_np(cm=val, weighted=weighted)
                    ax.plot(ema_dates, ema_vals, color=color, label=(leg_text_ema % val))

            if hl:
                for val, color in zip(hl, col_palette):
                    ema_vals = self.calc_ema_np(hl=val, weighted=weighted)
                    ax.plot(ema_dates, ema_vals, color=color, label=(leg_text_ema % val))

            if theta:
                for val, color in zip(theta, col_palette):
                    ema_vals = self.calc_ema_np(theta=val, weighted=weighted)
                    ax.plot(ema_dates, ema_vals, color=color, label=(leg_text_ema % val))
                    
        if win_boll:
            for val, col in zip()
                band_color = color_schemes[win_boll]['band']
            sma_vals, upper_band, lower_band, sma_dates = self.calc_boll_np(win_boll)
            ax.plot(sma_dates, lower_band, color=band_color, linestyle='--') # lower band
            ax.plot(sma_dates, sma_vals,   color=band_color, label='SMA (%d-day)' % win_boll) # sma
            ax.plot(sma_dates, upper_band, color=band_color, linestyle='-.')   # upper band
            ax.fill_between(sma_dates, lower_band, upper_band, color=color_schemes[win_boll]['fill'], label='Bollinger Band ($\pm%.1f\sigma$)' % band_legend[win_boll], alpha=0.2, zorder=2)

        ax.legend()
        fig.tight_layout()
        plt.show()

        if save_fig:
            if path is not None:
                path = format_path(path)
            else:
                path = ''
            
            plot_name = '%s_close_price' % self.ticker # declare base name for plot
            if win_boll:
                plot_name += '_bollinger_%d' % win_boll
                    
            if sma or ema:
                if sma:
                    avg_type = '_sma_' 
                    for win, cnt in zip(window, np.arange(len(window))):
                        if cnt < (len(window) - 1):
                            avg_type += '%d_' % win
                        else:
                            avg_type += '%d' % win
                    plot_name += avg_type
                
                if ema:
                    avg_type = '_ema_'
                    if span:
                        sp_ind_vars = span  # transfer the list of smoothing parameter independent values to general list
                        avg_type += 'span_' # add 'span' to the plot name
                    if cm:
                        sp_ind_vars = cm
                        avg_type += 'cm_'
                    if hl:
                        sp_ind_vars = hl
                        avg_type += 'hl_'
                    if theta:
                        sp_ind_vars = theta
                        avg_type += 'theta_'
                    
                    for val, cnt in zip(sp_ind_vars, np.arange(len(sp_ind_vars))):
                        if cnt < (len(sp_ind_vars) - 1):
                            avg_type += '%d_' % val
                        else:
                            avg_type += '%d.pdf' % val
                            
                    plot_name += avg_type
                else: # if no ema, add file extension
                    plot_name += '.pdf'
            else: # if no ema or sma, add file extension to plot name
                plot_name += '.pdf'

            fig.savefig(path + plot_name)  