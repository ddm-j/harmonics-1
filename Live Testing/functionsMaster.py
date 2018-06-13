import pandas as pd
import numpy as np
from scipy import stats
import scipy.optimize
from scipy.optimize import OptimizeWarning
import warnings
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_finance import candlestick_ohlc
from matplotlib.dates import date2num
from datetime import datetime

if 0:

    data = pd.read_csv('AUDUSD.csv')
    data.columns=[['Date','Ask','Bid','AskVol','BidVol']]
    data['Symbol'] = 'SYMB'
    data = data.set_index(pd.to_datetime(data['Date']))

    prices = data.copy()

if 0:

    data = pd.read_csv('AUDUSD10Min.csv')
    data = data.set_index(pd.to_datetime(data['Date']))

    prices = data.copy()

class holder:
    1



# Data Resampling Function

def OHLCresample(DataFrame,TimeFrame,column='ask'):

    """

    :param DataFrame: a pandas DataFrame containing FOREX bid/ask data as well as datetime() objects
    :param column: the data column for which to calculate OHLC data (bid or ask). Default = 'ask'
    :param TimeFrame: the timeframe for which to parse the data. Datatype: string. ie, 1Min, 5Min, 15Min, 30Min, 1Hr, 4Hr, 1D.....
    :return: resampled: OHLC data for the given timeframe, as well as the EMA & SMA over that timeframe

    """

    grouped = DataFrame.groupby('Symbol')

    if np.any(DataFrame.columns=='Ask'):

        if column == 'ask':

            ask = grouped['Ask'].resample(TimeFrame).ohlc()
            askVol = grouped['AskVol'].resample(TimeFrame).count()

            resampled = pd.DataFrame(ask)
            resampled['AskVol'] = askVol

        elif column == 'bid':

            bid = grouped['Bid'].resample(TimeFrame).ohlc()
            bidVol = grouped['BidVol'].resample(TimeFrame).count()

            resampled = pd.DataFrame(bid)
            resampled['BidVol'] = bidVol

        else:

            raise ValueError('Column must be a string. Either ask or bid.')

    elif np.any(DataFrame.columns=='close'):

        open = grouped['open'].resample(TimeFrame).ohlc()
        close = grouped['close'].resample(TimeFrame).ohlc()
        high = grouped['high'].resample(TimeFrame).ohlc()
        low = grouped['low'].resample(TimeFrame).ohlc()
        askVol = grouped['AskVol'].resample(TimeFrame).count()

        resampled = pd.DataFrame(open)
        resampled['high'] = high
        resampled['low'] = low
        resampled['close'] = close
        resampled['AskVol'] = askVol

    resampled = resampled.dropna()

    return resampled

# Momentum Function

def momentum(prices,periods):

    """

    :param prices: pandas dataframe containing bid and ask data
    :param periods: array like, containing integer periods for which to calculate the momentum
    :return: momentum for each of the periods


    """

    results = holder()

    open = {}
    close = {}

    for i in range(0,len(periods)):

        open[periods[i]] = pd.DataFrame(prices.close.iloc[periods[i]:]-prices.close.iloc[:-periods[i]].values
                                        ,index=prices.iloc[periods[i]:].index)
        close[periods[i]] = pd.DataFrame(prices.open.iloc[periods[i]:]-prices.open.iloc[:-periods[i]].values
                                         ,index=prices.iloc[periods[i]:].index)

        if np.any(close[periods[i]].index.duplicated()):
            print("DUPES!", close[periods[i]].iloc[np.where(close[periods[i]].index.duplicated())])

        open[periods[i]].columns = [['open']]
        close[periods[i]].columns = [['close']]

        #print(close[periods[i]][close[periods[i].index.duplicated()]])

    results.open = open
    results.close = close

    return results

# Stochastic Oscillator Function

def stochastic(prices,periods):

    """

    :param prices: dataframe containing prices
    :param periods: periods for which to calculate the stochastic oscillator
    :return: stochastic oscillator for each of the periods


    """

    results = holder()

    close = {}

    for i in range(0,len(periods)):

        Ks = []

        for j in range(periods[i],len(prices)-periods[i]):

            C = prices.close.iloc[j+1]
            H = prices.high.iloc[j-periods[i]:j].max()
            L = prices.low.iloc[j-periods[i]:j].min()

            if H == L:

                K = 0

            else:

                K = 100*(C-L)/(H-L)

            Ks = np.append(Ks,K)

        df = pd.DataFrame(Ks,index=prices.iloc[periods[i]+1:-periods[i]+1].index)
        df.columns = [['K']]
        df['D'] = df.K.rolling(3).mean()
        df = df.dropna()

        close[periods[i]] = df

    results.close = close

    return results


# Williams Oscillator Function

def williams(prices,periods):

    """

    :param prices: dataframe containing prices
    :param periods: array like containing periods for which to calculate the prices
    :return: williams oscillator


    """

    results = holder()

    close = {}

    for i in range(0,len(periods)):

        Rs = []

        for j in range(periods[i],len(prices)-periods[i]):

            C = prices.close.iloc[j+1]
            H = prices.high.iloc[j-periods[i]:j].max()
            L = prices.low.iloc[j-periods[i]:j].min()

            if H == L:

                R = 0

            else:

                R = -100 * (H - C) / (H - L)

            Rs = np.append(Rs,R)

        df = pd.DataFrame(Rs,index=prices.iloc[periods[i]+1:-periods[i]+1].index)
        df.columns = [['R']]
        df = df.dropna()

        close[periods[i]] = df

    results.close = close

    return results


# PROC Function (Price Rate of Change)

def proc(prices,periods):

    """

    :param prices: dataframe containing prices
    :param periods: array like containing list of periods for which to calculate PROC
    :return: PROC for indicated periods


    """

    results = holder()

    proc = {}

    for i in range(0,len(periods)):

        proc[periods[i]] = pd.DataFrame((prices.close.iloc[periods[i]:]-prices.close.iloc[:-periods[i]].values)/prices.close.iloc[:-periods[i]].values)
        proc[periods[i]].columns = [['close']]

    results.proc = proc



    return results


# Accumulation Distribution Oscillator

def adosc(prices,periods):

    """

    :param prices: dataframe containing prices
    :param periods: periods for which to calculate the ADOSC
    :return: ADOSC


    """

    results = holder()

    accdist = {}

    for i in range(0,len(periods)):

        AD = []

        for j in range(periods[i],len(prices)-periods[i]):

            C = prices.close.iloc[j+1]
            H = prices.high.iloc[j-periods[i]:j].max()
            L = prices.low.iloc[j-periods[i]:j].min()
            V = prices.AskVol.iloc[j+1]

            if H == L:

                CLV = 0

            else:

                CLV = ((C-L)-(H-C))/(H-L)

            AD = np.append(AD,CLV*V)

        AD = AD.cumsum()
        AD = pd.DataFrame(AD,index=prices.iloc[periods[i]+1:-periods[i]+1].index)
        AD.columns = [['AD']]
        accdist[periods[i]] = AD

    results.AD = accdist

    return results


# MACD (Moving Average Convergence Divergence

def macd(prices,periods):

    """

    :param prices: dataframe containing prices
    :param periods: 1x2 array containing values for the EMAs.
    :return: MACD for given periods


    """

    results = holder()

    EMA1 = prices.close.ewm(span=periods[0]).mean()
    EMA2 = prices.close.ewm(span=periods[1]).mean()

    MACD = pd.DataFrame(EMA1 - EMA2)
    MACD.columns = [['L']]
    SigMACD = MACD.rolling(3).mean()
    SigMACD.columns = [['SL']]

    results.line = MACD
    results.signal = SigMACD

    return results

# Commodity Channel Index

def cci(prices,periods):

    """


    :param prices: dataframe containing prices
    :param periods: array containing periods for which to calculate the CCI
    :return: CCI for periods


    """

    results = holder()

    CCI = {}

    for i in range(0,len(periods)):

        MA = prices.close.rolling(periods[i]).mean()
        std = prices.close.rolling(periods[i]).std()

        D = (prices.close-MA)/std

        CCI[periods[i]] = pd.DataFrame((prices.close - MA)/(0.015*D))
        CCI[periods[i]].columns = [['close']]


    results.cci = CCI

    return results


# Bollinger Bands

def bollinger(prices,periods,deviations):

    """


    :param prices: dataframe containing prices
    :param periods: periouds for which to calculate bollinger bands
    :param deviations: deviations to use when calculating bands
    :return: bollinger bands



    """

    results = holder()

    boll = {}

    for i in range(0,len(periods)):

        mid = prices.close.rolling(periods[i]).mean()
        std = prices.close.rolling(periods[i]).std()

        upper = mid+deviations*std
        lower = mid-deviations*std

        df = pd.concat((upper,mid,lower),axis=1)
        df.columns = [['upper','mid','lower']]

        boll[periods[i]] = df

    results.bands = boll

    return results


# Heiken Ashi Candles

def heikenashi(prices,periods):

    """


    :param prices: dataframe of OHLC prices
    :param periods: periods for which to get the heiken ashi candles
    :return: heiken ashi candles


    """

    results = holder()

    dict = {}

    HAclose = prices[['open','high','low','close']].sum(axis=1)/4

    HAopen = HAclose.copy()
    HAopen.iloc[0] = HAclose.iloc[0]

    HAhigh = HAclose.copy()

    HAlow = HAclose.copy()

    for i in range(1,len(prices)):

        HAopen.iloc[i] = (HAopen.iloc[i-1]+HAclose.iloc[i-1])/2
        HAhigh.iloc[i] = np.array([prices.high.iloc[i],HAopen.iloc[i],HAclose.iloc[i]]).max()
        HAlow.iloc[i] = np.array([prices.low.iloc[i],HAopen.iloc[i],HAopen.iloc[i]]).min()

    df = pd.concat((HAopen,HAhigh,HAlow,HAclose),axis=1)
    df.columns = [['open','high','low','close']]
    df.index = df.index.droplevel(0)

    dict[periods[0]] = df

    results.candles = dict

    return results

# Price Averages!

def paverage(prices,periods):

    """

    :param prices: dataframe of prices
    :param periods: periods for which to calculate price averages
    :return: averages over the given periods


    """

    results = holder()

    avs = {}

    for i in range(0,len(periods)):

        avs[periods[i]] = pd.DataFrame(prices[['open','high','low','close']].rolling(periods[i]).mean())

    results.avs = avs

    return results

# Slopes

def slopes(prices,periods):

    """

    :param prices: dataframe containing prices
    :param periods: periods for which to calculate slopes
    :return: slopes over those periods


    """

    results = holder()

    slope = {}

    for i in range(0,len(periods)):

        ms = []

        for j in range(periods[i],len(prices)-periods[i]):

            y = prices.high.iloc[j-periods[i]:j].values
            x = np.arange(0,len(y))

            res = stats.linregress(x,y=y)
            m = res.slope

            ms = np.append(ms,m)

        ms = pd.DataFrame(ms,index=prices.iloc[periods[i]:-periods[i]].index)

        ms.columns = [['high']]

        slope[periods[i]] = ms

    results.slope = slope

    return results


# Detrender

def detrend(prices,method='difference'):

    """

    :param prices: dataframe of prices to remove the trend from
    :param method: 'linear' - assumes linear trend and uses linear regression over the time series to remove the trend
                    'difference' - removes the trend using the difference between the current price and the last price
    :return: dataframe of price that has been trend cleansed
    """

    #Detrend the data

    if method == 'difference':

        detrended = prices.close[1:]-prices.close[:-1].values

    elif method == 'linear':

        x = np.arange(0,len(prices))
        y = prices.close.values

        model = LinearRegression()
        model.fit(x.reshape((-1,1)),y.reshape((-1,1)))

        trend = model.predict(x.reshape((-1,1)))

        trend = trend.reshape((144,))

        detrended = prices.close - trend

    else:

        print('A valid method was not entered. Options are: difference, linear')

    return detrended

# Fourier Series Expansion fitting function

def fseries(x,a0,a1,b1,w):

    """

    :param x: x-data to evaluate the function at
    :param a0: the first fourier expansion constant
    :param a1: the first cosine fourier expansion coefficient
    :param b1: the first sine fourier expansion coefficient
    :param w: the frequency
    :return: evaluation of the fourier expansion


    """

    f = a0 + a1*np.cos(x*w) + b1*np.sin(x*w)

    return f

# Fourier Series Coefficient function

def fourier(prices,periods,method='difference'):

    """

    :param prices: dataframe object containing prices
    :param periods: periods for which to get the fourier expansion fitting coefficients
    :param method: method used to detrend the time series. See 'detrend' above
    :return: array containing the fourier coefficients for each time section


    """

    results = holder()

    dict = {}

    # Option to plot the expansion fit for each iteration

    plot = 'False'

    # Compute coefficients of the series

    detrended = detrend(prices,method)

    for i in range(0,len(periods)):

        coeffs = []

        for j in range(periods[i],len(prices)-periods[i]):

            x = np.arange(0,periods[i])
            y = detrended.iloc[j-periods[i]:j].values

            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)

                try:
                    res = scipy.optimize.curve_fit(fseries, x, y)

                except (RuntimeError,OptimizeWarning):
                    res = np.empty((1,4))
                    res[0,:] = np.NAN


            if plot == 'True':
                xt = np.linspace(0, periods[i], 100)
                yt = fseries(xt,res[0][0],res[0][1],res[0][2],res[0][3])
                plt.plot(x,y)
                plt.plot(xt,yt,'r')
                plt.show()


            coeffs = np.append(coeffs,res[0],axis=0)

        warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        coeffs = np.array(coeffs).reshape(((len(coeffs)/4,4)))
        df = pd.DataFrame(coeffs,index=prices.iloc[periods[i]:-periods[i]].index)
        df.columns = [['a0','a1','b1','w']]
        df = df.fillna(method='bfill')

        dict[periods[i]] = df

    results.coeffs = dict

    return results

# Sine Series Fitting Function

def sseries(x,a0,b1,w):

    f = a0 + b1*np.sin(x*w)

    return f

# Sine Series Coefficients Function

def sine(prices,periods,method='difference'):

    """

    :param prices: dataframe object containing prices
    :param periods: periods for which to get the sine expansion fitting coefficients
    :param method: method used to detrend the time series. See 'detrend' above
    :return: array containing the sine fitting coefficients for each time section


    """

    results = holder()

    dict = {}

    # Option to plot the expansion fit for each iteration

    plot = 'False'

    # Compute coefficients of the series

    detrended = detrend(prices,method)

    for i in range(0,len(periods)):

        coeffs = []

        for j in range(periods[i],len(prices)-periods[i]):

            x = np.arange(0,periods[i])
            y = detrended.iloc[j-periods[i]:j].values

            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)

                try:
                    res = scipy.optimize.curve_fit(sseries, x, y)

                except (RuntimeError,OptimizeWarning):
                    res = np.empty((1,3))
                    res[0,:] = np.NAN


            if plot == 'True':
                xt = np.linspace(0, periods[i], 100)
                yt = sseries(xt,res[0][0],res[0][1],res[0][2])
                plt.plot(x,y)
                plt.plot(xt,yt,'r')
                plt.show()


            coeffs = np.append(coeffs,res[0],axis=0)



        warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        coeffs = np.array(coeffs).reshape(((len(coeffs)/3,3)))
        df = pd.DataFrame(coeffs,index=prices.iloc[periods[i]:-periods[i]].index)
        df.columns = [['a0','b1','w']]
        df = df.fillna(method='bfill')

        dict[periods[i]] = df

    results.coeffs = dict

    return results

# Williams Accumulation Distribution Function

def wadl(prices,periods):

    results = holder()
    dict = {}

    for i in range(0,len(periods)):

        WAD = []

        for j in range(periods[i],len(prices)-periods[i]):

            TRH = np.array([prices.high.iloc[j],prices.close.iloc[j-1]]).max()
            TRL = np.array([prices.low.iloc[j],prices.close.iloc[j-1]]).min()

            if prices.close.iloc[j]>prices.close.iloc[j-1]:

                PM = prices.close.iloc[j] - TRL

            elif prices.close.iloc[j]<prices.close.iloc[j-1]:

                PM = prices.close.iloc[j] - TRH

            elif prices.close.iloc[j] == prices.close.iloc[j-1]:

                PM = 0

            AD = PM * prices.AskVol.iloc[j]

            WAD = np.append(WAD,AD)

        WAD = WAD.cumsum()
        WAD = pd.DataFrame(WAD,index = prices.iloc[periods[i]:-periods[i]].index)
        WAD.columns = [['close']]

        dict[periods[i]] = WAD

    results.wadl = dict

    return results

# Position Sizing Function

def posSize(accountBalance,percentageRisk,pipRisk,rate):

    if rate > 0:

        rate = 1/rate

    trade = (percentageRisk/100)*accountBalance

    pipval = trade/pipRisk

    size = pipval*10000

    return size
