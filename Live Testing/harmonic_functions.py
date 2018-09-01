import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, argrelmax, argrelmin
import fast_any_all as faa

def is_pattern(fib_levels,labels,moves,err_allowed):

    conditions = [generate_cond(fib_levels['gartley'],moves,err_allowed),
                  generate_cond(fib_levels['butterfly'],moves,err_allowed),
                  generate_cond(fib_levels['bat'],moves,err_allowed),
                  generate_cond(fib_levels['crab'],moves,err_allowed)]

    if faa.any(conditions):
        return [labels[np.where(conditions)[0][0]],'Bearish' if moves[0] < 0 else 'Bullish']
    else:
        return None

def generate_cond(levels, moves, err_allowed):
    cond = levels[0][0]-err_allowed < abs(moves[1]) < levels[0][1]+err_allowed and \
           levels[1][0]-err_allowed < abs(moves[2]) < levels[1][1]+err_allowed and \
           levels[2][0]-err_allowed < abs(moves[3]) < levels[2][1]+err_allowed
    return cond

def is_gartley(moves,err_allowed):

    err_allowed = float(err_allowed)/float(100)

    AB_ret = np.array([61.8 / 100 - err_allowed, 61.8 / 100 + err_allowed]) * abs(moves[0])

    BC_ret = np.array([38.2 / 100 - err_allowed, 88.6 / 100 + err_allowed]) * abs(moves[1])

    CD_ret = np.array([1.27 - err_allowed, 1.618 + err_allowed]) * abs(moves[2])

    if AB_ret[0] < abs(moves[1]) < AB_ret[1] and BC_ret[0] < abs(moves[2]) < BC_ret[1] and CD_ret[0] < abs(moves[3]) < CD_ret[1]:

        return 1 if moves[0] > 0 else -1

    else:
        return np.NAN


def is_butterfly(moves,err_allowed):

    err_allowed = float(err_allowed)/float(100)

    AB_ret = np.array([78.6 / 100 - err_allowed, 78.6 / 100 + err_allowed]) * abs(moves[0])

    BC_ret = np.array([38.2 / 100 - err_allowed, 88.6 / 100 + err_allowed]) * abs(moves[1])

    CD_ret = np.array([1.618 - err_allowed, 2.618 + err_allowed]) * abs(moves[2])

    if AB_ret[0] < abs(moves[1]) < AB_ret[1] and BC_ret[0] < abs(moves[2]) < BC_ret[1] and CD_ret[0] < abs(moves[3]) < CD_ret[1]:

        return 1 if moves[0] > 0 else -1

    else:

        return np.NAN


def is_bat(moves,err_allowed):

    err_allowed = float(err_allowed)/float(100)

    AB_ret = np.array([32.8 / 100 - err_allowed, 50.0 / 100 + err_allowed]) * abs(moves[0])

    BC_ret = np.array([38.2 / 100 - err_allowed, 88.6 / 100 + err_allowed]) * abs(moves[1])

    CD_ret = np.array([161.8/100 - err_allowed, 261.8/100 + err_allowed]) * abs(moves[2])

    if AB_ret[0] < abs(moves[1]) < AB_ret[1] and BC_ret[0] < abs(moves[2]) < BC_ret[1] and CD_ret[0] < abs(moves[3]) < CD_ret[1]:

        return 1 if moves[0] > 0 else -1

    else:

        return np.NAN


def is_crab(moves,err_allowed):

    err_allowed = float(err_allowed)/float(100)

    AB_ret = np.array([38.2 / 100 - err_allowed, 61.8 / 100 + err_allowed]) * abs(moves[0])

    BC_ret = np.array([38.2 / 100 - err_allowed, 88.6 / 100 + err_allowed]) * abs(moves[1])

    CD_ret = np.array([224.0 / 100 - err_allowed, 361.8 / 100 + err_allowed]) * abs(moves[2])

    if AB_ret[0] < abs(moves[1]) < AB_ret[1] and BC_ret[0] < abs(moves[2]) < BC_ret[1] and CD_ret[0] < abs(moves[3]) < CD_ret[1]:

        return 1 if moves[0] > 0 else -1

    else:

        return np.NAN


def is_shark(moves,err_allowed):

    err_allowed = float(err_allowed)/float(100)

    BC_ret = np.array([113.0 / 100 - err_allowed, 161.8 / 100 + err_allowed]) * abs(moves[0])

    CD_ret = np.array([161.8 / 100 - err_allowed, 224.0 / 100 + err_allowed]) * abs(moves[1])


    if BC_ret[0] < abs(moves[2]) < BC_ret[1] and CD_ret[0] < abs(moves[3]) < CD_ret[1]:

        return 1 if moves[0] > 0 else -1

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

def fft_detect(price, p=0.4):

    trans = np.fft.rfft(price)
    trans[int(p*len(trans)):] = 0
    inv = np.fft.irfft(trans)
    dy = np.gradient(inv)
    patt_idx = (np.where(np.diff(np.sign(dy)))[0] + 1)[1:]

    label = np.array([x for x in np.diff(np.sign(dy)) if x != 0])

    # Look for Better Peaks

    if 0.1 <= p < 0.2:
        l = 3
    elif p == 0.2:
        l = 3
    elif 0.2 < p <= 0.3:
        l = 2
    elif p > 0.3:
        l = 2

    # Define the bounds beforehand, its marginally faster than doing it in the loop
    upper = np.array(patt_idx) + (l+1)
    lower = np.array(patt_idx) - (l-1)

    # List comprehension...
    new_inds = [price[low:hi].argmax()+low if lab == 2 else
                price[low:hi].argmin()+low
                for low, hi, lab in zip(lower, upper, label)]

    new_inds = np.append(new_inds, len(price) - 1)

    #plt.plot(price)
    #plt.plot(inv)
    #plt.scatter(new_inds,price[new_inds])
    #plt.scatter(patt_idx,price[patt_idx],c='r')
    #plt.show()

    return new_inds, price[new_inds]

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

def posSizeBT(accountBalance,percentageRisk,pipRisk,u_j=False,quote=None):

    trade = (percentageRisk/100)*accountBalance

    if u_j:

        trade *= quote

    pipval = trade/pipRisk

    size = pipval*100 if u_j else pipval*10000 # 10000 units / (1$/pip)

    return size
