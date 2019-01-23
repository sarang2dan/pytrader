import numpy as np
import pandas as pd
import pandas.io.data as web
from pandas.compat import range, lrange, lmap, map, zip
from pandas.tools.plotting import scatter_matrix,autocorrelation_plot
import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter

# todo: 분활된 화면에 출력하는 모듈로 발전시켰으면 좋겠다.
# todo: 그래프의 출력은 Realtime Trading Monitor로 발전시키는 것이 최종목표..
class ChartDrawer:

    def drawStationarityTestHistogram(df):
        fig, axs = plt.subplots(3, 1)

        df['adf_5'].plot(kind='hist', title="ADF 5%", ax=axs[0])
        df['hurst_exp'].plot(kind='hist', title="Hurst Exponent", ax=axs[1])
        df['halflife'].plot(kind='hist', title="Half Life", ax=axs[2])

        plt.show()


    def drawStationarityRankHistogram(df):
        fig, axs = plt.subplots(3,1)
        df['rank_adf'].plot(kind='hist', title="ADF", ax=axs[0])
        df['rank_hurst'].plot(kind='hist', title="Hurst Exponent", ax=axs[1])
        df['rank_halflife'].plot(kind='hist', title="Half Life", ax=axs[2])
        plt.show()
        return

    def drawStationarityTestBoxPlot(df):
        df_st = df[['adf_5', 'hurst', 'halflife']]
        df_st.plot(kind='box',
                   layout=(1, 3),
                   subplots=True,
                   title="Stationarity Test")
        plt.show()
        return

    def plot_price_series(df, ts1, ts2):
        months = MonthLocator()  # every month
        fig, ax = plt.subplots()
        ax.plot(df.index, df[ts1], label=ts1)
        ax.plot(df.index, df[ts2], label=ts2)
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
        ax.set_xlim(datetime.datetime(2012, 1, 1), datetime.datetime(2013, 1, 1))
        ax.grid(True)
        fig.autofmt_xdate()

        plt.xlabel('Month/Year')
        plt.ylabel('Price ($)')
        plt.title('%s and %s Daily Prices' % (ts1, ts2))
        plt.legend()
        plt.show()
        return

    def plot_scatter_series(df, ts1, ts2):
        plt.xlabel('%s Price ($)' % ts1)
        plt.ylabel('%s Price ($)' % ts2)
        plt.title('%s and %s Price Scatterplot' % (ts1, ts2))
        plt.scatter(df[ts1], df[ts2])
        plt.show()
        return

    def plot_residuals(df):
        months = MonthLocator()  # every month
        fig, ax = plt.subplots()
        ax.plot(df.index, df["res"], label="Residuals")
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
        ax.set_xlim(datetime.datetime(2012, 1, 1), datetime.datetime(2013, 1, 1))
        ax.grid(True)
        fig.autofmt_xdate()

        plt.xlabel('Month/Year')
        plt.ylabel('Price ($)')
        plt.title('Residual Plot')
        plt.legend()

        plt.plot(df["res"])
        plt.show()
        return