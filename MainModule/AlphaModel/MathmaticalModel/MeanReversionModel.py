from MainModule.AlphaModel.AlphaModel import AlphaModel
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from logger import logger
from MainModule.DataTypes import *
import statsmodels.tsa.stattools as ts

class MeanReversionModel(AlphaModel):
    def __init__(self, window_size, threshold):
        self.window_size = window_size
        self.threshold = threshold
        return

    def calcADF(self, df: pd.DataFrame):
        adf_result = ts.adfuller(df)
        adf_p_value = adf_result[4]

        # adf_result[0] is adf_statistics
        return adf_result[0], adf_p_value['1%'], adf_p_value['5%'], adf_p_value['10%']

    def hurst(self, ts):
        """Returns the Hurst Exponent of the time series vector ts"""
        # Create the range of lag values
        lags = range(2, 100)

        # Calculate the array of the variances of the lagged differences
        tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

        # Use a linear fit to estimate the Hurst Exponent
        poly = polyfit(log(lags), log(tau), 1)

        # Return the Hurst exponent from the polyfit output
        return poly[0]*2.0

    def calcHurstExponent(self, df, lags_count=100):
        lags = range(2, lags_count)
        
        ts = np.log(df)

        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]

        poly = np.polyfit(np.log(lags), np.log(tau), 1)

        result = poly[0] * 2.0
        return result

    def calcHalfLife(self, df):
        price = Series(df)
        lagged_price = price.shift(1).fillna(method='bfill')
        delta = price - lagged_price
        beta = np.polyfit(lagged_price, delta, 1)[0]
        half_life = (-1 * (np.log(2) / beta))
        return half_life

    def determinePosition(self, df, col_name, row_index, verbose=False):
        cur_row_value = df.ix[row_index][col_name]

        df_col = df.ix[0:row_index][col_name]

        df_moving_avg = df_col.rolling(self.window_size).mean()
        df_moving_avg_std = df_col.rolling(self.window_size).std()

        moving_avg = df_moving_avg[row_index]
        moving_avg_std = df_moving_avg_std[row_index]

        price_arbitrage = cur_row_value - moving_avg

        if verbose == True:
            logger.info("[diff: %s][%s: %s][moving_avg: %s][moving_avg_std: %s]" % \
                         (str(price_arbitrage),
                          col_name,
                          str(cur_row_value),
                          str(moving_avg),
                          str(moving_avg_std)))

        if abs(price_arbitrage) > moving_avg * self.threshold:
            if np.sign(price_arbitrage) > 0:
                return POSITION.SHORT # sell
            else:
                return POSITION.LONG  # BUY

        return POSITION.HOLD
