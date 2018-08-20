import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

def is_gartley(moves,err_allowed):

    err_allowed = float(err_allowed)/float(100)

    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]

    AB_ret = np.array([61.8 / 100 - err_allowed, 61.8 / 100 + err_allowed]) * abs(XA)

    BC_ret = np.array([38.2 / 100 - err_allowed, 88.6 / 100 + err_allowed]) * abs(AB)

    CD_ret = np.array([1.27 - err_allowed, 1.618 + err_allowed]) * abs(BC)


    if XA > 0 and AB < 0 and BC > 0 and CD < 0:

        # Bullish Gartley

        if AB_ret[0] < abs(AB) < AB_ret[1] and BC_ret[0] < abs(BC) < BC_ret[1] and CD_ret[0] < abs(CD) < CD_ret[1]:

            return 1

        else:
            return np.NAN

    elif XA < 0 and AB > 0 and BC < 0 and CD > 0:

        # Bearish Gartley

        if AB_ret[0] < abs(AB) < AB_ret[1] and BC_ret[0] < abs(BC) < BC_ret[1] and CD_ret[0] < abs(CD) < CD_ret[1]:

            return -1

        else:
            return np.NAN


    else:

        return np.NAN


def is_butterfly(moves,err_allowed):

    err_allowed = float(err_allowed)/float(100)

    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]

    AB_ret = np.array([78.6 / 100 - err_allowed, 78.6 / 100 + err_allowed]) * abs(XA)

    BC_ret = np.array([38.2 / 100 - err_allowed, 88.6 / 100 + err_allowed]) * abs(AB)

    CD_ret = np.array([1.618 - err_allowed, 2.618 + err_allowed]) * abs(BC)

    if XA > 0 and AB < 0 and BC > 0 and CD < 0:

        # Bullish Butterfly

        if AB_ret[0] < abs(AB) < AB_ret[1] and BC_ret[0] < abs(BC) < BC_ret[1] and CD_ret[0] < abs(CD) < CD_ret[1]:

            return 1

        else:

            return np.NAN

    elif XA < 0 and AB > 0 and BC < 0 and CD > 0:

        # Bearish Butterfly

        if AB_ret[0] < abs(AB) < AB_ret[1] and BC_ret[0] < abs(BC) < BC_ret[1] and CD_ret[0] < abs(CD) < CD_ret[1]:

            return -1

        else:

            return np.NAN

    else:

        return np.NAN


def is_bat(moves,err_allowed):

    err_allowed = float(err_allowed)/float(100)

    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]

    AB_ret = np.array([32.8 / 100 - err_allowed, 50.0 / 100 + err_allowed]) * abs(XA)

    BC_ret = np.array([38.2 / 100 - err_allowed, 88.6 / 100 + err_allowed]) * abs(AB)

    CD_ret = np.array([161.8/100 - err_allowed, 261.8/100 + err_allowed]) * abs(BC)


    if XA > 0 and AB < 0 and BC > 0 and CD < 0:

        # Bullish Butterfly

        if AB_ret[0] < abs(AB) < AB_ret[1] and BC_ret[0] < abs(BC) < BC_ret[1] and CD_ret[0] < abs(CD) < CD_ret[1]:

            return 1

        else:

            return np.NAN

    elif XA < 0 and AB > 0 and BC < 0 and CD > 0:

        # Bearish Butterfly

        if AB_ret[0] < abs(AB) < AB_ret[1] and BC_ret[0] < abs(BC) < BC_ret[1] and CD_ret[0] < abs(CD) < CD_ret[1]:

            return -1

        else:

            return np.NAN

    else:

        return np.NAN


def is_crab(moves,err_allowed):

    err_allowed = float(err_allowed)/float(100)

    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]

    AB_ret = np.array([38.2 / 100 - err_allowed, 61.8 / 100 + err_allowed]) * abs(XA)

    BC_ret = np.array([38.2 / 100 - err_allowed, 88.6 / 100 + err_allowed]) * abs(AB)

    CD_ret = np.array([224.0 / 100 - err_allowed, 361.8 / 100 + err_allowed]) * abs(BC)

    if XA > 0 and AB < 0 and BC > 0 and CD < 0:

        # Bullish Butterfly

        if AB_ret[0] < abs(AB) < AB_ret[1] and BC_ret[0] < abs(BC) < BC_ret[1] and CD_ret[0] < abs(CD) < CD_ret[1]:

            return 1

        else:

            return np.NAN

    elif XA < 0 and AB > 0 and BC < 0 and CD > 0:

        # Bearish Butterfly

        if AB_ret[0] < abs(AB) < AB_ret[1] and BC_ret[0] < abs(BC) < BC_ret[1] and CD_ret[0] < abs(CD) < CD_ret[1]:

            return -1

        else:

            return np.NAN

    else:

        return np.NAN


def is_shark(moves,err_allowed):

    err_allowed = float(err_allowed)/float(100)

    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]

    BC_ret = np.array([113.0 / 100 - err_allowed, 161.8 / 100 + err_allowed]) * abs(XA)

    CD_ret = np.array([161.8 / 100 - err_allowed, 224.0 / 100 + err_allowed]) * abs(AB)

    if XA > 0 and AB < 0 and BC > 0 and CD < 0:

        # Bullish Butterfly

        if BC_ret[0] < abs(BC) < BC_ret[1] and CD_ret[0] < abs(CD) < CD_ret[1]:

            return 1

        else:

            return np.NAN

    elif XA < 0 and AB > 0 and BC < 0 and CD > 0:

        # Bearish Butterfly

        if BC_ret[0] < abs(BC) < BC_ret[1] and CD_ret[0] < abs(CD) < CD_ret[1]:

            return -1

        else:

            return np.NAN

    else:

        return np.NAN



def peak_detect(price,peak_range=5):

    max_idx = list(argrelextrema(price, np.greater, order=peak_range)[0])
    min_idx = list(argrelextrema(price, np.less, order=peak_range)[0])

    max = price[max_idx]
    min = price[min_idx]

    peaks_idx = max_idx + min_idx

    peaks_idx = np.append(peaks_idx,len(price)-1)

    peaks_idx.sort()

    peaks = price[peaks_idx]

    return peaks_idx,peaks


# Position Sizing Function

def posSize(accountBalance,percentageRisk,pipRisk,rate):

    if rate > 0:

        rate = 1/rate

    trade = (percentageRisk/100)*accountBalance

    pipval = trade/pipRisk

    size = pipval*10000

    return size


def walk_forward(price,sign,slippage=4,stop=10):

    slippage = float(slippage)/float(10000)
    stop_amount = float(stop)/float(10000)

    if sign == 1:

        initial_stop_loss = price[0] - stop_amount

        stop_loss = initial_stop_loss

        for i in range(1,len(price)):

            move = price[i] - price[i-1]

            if move > 0 and (price[i]-stop_amount) > initial_stop_loss:

                stop_loss = price[i] - stop_amount

            elif price[i] < stop_loss:

                return stop_loss - price[0] - slippage

    elif sign == -1:

        initial_stop_loss = price[0] + stop_amount

        stop_loss = initial_stop_loss

        for i in range(1, len(price)):

            move = price[i] - price[i-1]

            if move < 0 and (price[i] + stop_amount) < initial_stop_loss:
                stop_loss = price[i] + stop_amount

            elif price[i] > stop_loss:

                return price[0] - stop_loss - slippage

def resampler(price,timeframe='M',length=30):

    if timeframe == 'M':

        frame = str(length) + 'T'

    else:

        frame = str(length) + 'H'


    price = price.resample(frame).ohlc()

    return price.close

# Position Sizing Function

def posSizeBT(accountBalance,percentageRisk,pipRisk):

    trade = (percentageRisk/100)*accountBalance

    pipval = trade/pipRisk

    size = pipval*10000 # 10000 units / (1$/pip)

    return size
