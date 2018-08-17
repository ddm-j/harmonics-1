import talib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

prices = pd.read_csv('ytd/EUR_USD.csv')
prices.columns = ['Date', 'open', 'high', 'low', 'close','volume']

prices.Date = pd.to_datetime(prices.Date, format='%d.%m.%Y %H:%M:%S.%f')

prices = prices.set_index(prices.Date)

prices = prices[['open', 'high', 'low', 'close', 'volume']]

prices = prices.drop_duplicates(keep=False)

prices = prices.iloc[100:201]

def get_indicators(prices):

    indicators = prices.copy()

    # Overlap Studies

    dema = talib.DEMA(prices.close,timeperiod=30)
    ht_line = talib.HT_TRENDLINE(prices.close)

    # Momentum Indicators

    adx = talib.ADX(prices.high,prices.low,prices.close,timeperiod=14)
    apo = talib.APO(prices.close, fastperiod=12, slowperiod=26, matype=0)
    bop = talib.BOP(prices.open,prices.high,prices.low,prices.close)
    cci = talib.CCI(prices.high,prices.low,prices.close,timeperiod=14)
    cmo = talib.CMO(prices.close,timeperiod=14)
    dx = talib.DX(prices.high,prices.low,prices.close,timeperiod=14)
    macd, macdsignal, macdhist = talib.MACD(prices.close, fastperiod=12, slowperiod=26, signalperiod=9)
    mfi = talib.MFI(prices.high, prices.low, prices.close, prices.volume, timeperiod=14)
    roc = talib.ROC(prices.close, timeperiod=10)
    rsi = talib.RSI(prices.close, timeperiod=14)
    slowk, slowd = talib.STOCH(prices.high, prices.low, prices.close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3,
                         slowd_matype=0)
    trix = talib.TRIX(prices.close, timeperiod=20)
    ultosc = talib.ULTOSC(prices.high, prices.low, prices.close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    willr = talib.WILLR(prices.high, prices.low, prices.close, timeperiod=14)

    # Volatility Indicators

    natr = talib.NATR(prices.high, prices.low, prices.close, timeperiod=14)

    # Volume Indicators

    adosc = talib.ADOSC(prices.high, prices.low, prices.close, prices.volume, fastperiod=3, slowperiod=10)
    obv = talib.OBV(prices.close,prices.volume)

    # Cycle Indicators

    ht_dcperiod = talib.HT_DCPERIOD(prices.close)
    ht_dcphase = talib.HT_DCPHASE(prices.close)
    inphase, quadrature = talib.HT_PHASOR(prices.close)
    sine, leadsine = talib.HT_SINE(prices.close)
    ht_trendmode = talib.HT_TRENDMODE(prices.close)

    ind_list = ['dema', 'ht_line', 'adx' , 'apo', 'bop', 'cci',
                'cmo', 'dx', 'macd', 'macdsignal', 'macdhist',
                'mfi', 'roc', 'rsi', 'slowk', 'slowd', 'trix',
                'ultosc', 'willr', 'natr', 'adosc', 'obv',
                'ht_dcperiod', 'ht_dcphase', 'inphase', 'quadrature',
                'sine', 'leadsine', 'ht_trendmode']
    data_list = [dema, ht_line, adx, apo, bop, cci,
                cmo, dx, macd, macdsignal, macdhist,
                mfi, roc, rsi, slowk, slowd, trix,
                ultosc, willr, natr, adosc, obv,
                ht_dcperiod, ht_dcphase, inphase, quadrature,
                sine, leadsine, ht_trendmode]

    for i in range(0,len(ind_list)):

        indicators[ind_list[i]] = data_list[i]

    print(indicators.iloc[-1])

    #plt.plot(prices.close)
    #plt.show()
    plt.plot(ht_trendmode)
    plt.show()

get_indicators(prices)

