import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

data = pd.read_csv('ytd/EUR_USD.csv')
data.columns = ['Date', 'open', 'high', 'low', 'close','volume']

data.Date = pd.to_datetime(data.Date, format='%d.%m.%Y %H:%M:%S.%f')

data = data.set_index(data.Date)

data = data[['open', 'high', 'low', 'close']]

data = data.drop_duplicates(keep=False)

price = data.close.values


def fft_detect(price, p=0.4,method='mine'):

    trans = np.fft.rfft(price)
    trans[round(p*len(trans)):] = 0
    inv = np.fft.irfft(trans)
    dy = np.gradient(inv)
    peaks_idx = np.where(np.diff(np.sign(dy)) == -2)[0] + 1
    valleys_idx = np.where(np.diff(np.sign(dy)) == 2)[0] + 1

    patt_idx = list(peaks_idx) + list(valleys_idx)
    patt_idx.sort()

    label = np.array([x for x in np.diff(np.sign(dy)) if x != 0])

    #peaks_idx = np.append(peaks_idx,len(price) - 1)

    # Look for Better Peaks

    l = 2

    new_inds = []


    if method=='first':
        search = np.vstack((np.arange(i - (l + 1), i + (l + 1)) for i in
                            patt_idx))
        price_mat = price.copy()[search]

        price_mat[label == -2] *= -1
        idx = np.argmin(price_mat, axis=1)

        new_inds = search[np.arange(idx.shape[0]), idx]

    elif method == 'mine':

        for i in range(0,len(patt_idx[:])):

            search = np.arange(patt_idx[i]-(l+1),patt_idx[i]+(l+1))

            if label[i] == -2:

                idx = price[search].argmax()

            elif label[i] == 2:

                idx = price[search].argmin()

            new_max = search[idx]

            new_inds.append(new_max)

            #plt.plot(price)
            #plt.scatter(search, price[search])
            #plt.scatter(i,price[i],c='r')
            #plt.scatter(new_max,price[new_max],c='g')
            #plt.show()

    elif method =='second':
        l = 2

        # Define the bounds beforehand, its marginally faster than doing it in the loop
        upper = np.array(patt_idx) + l + 1
        lower = np.array(patt_idx) - l - 1

        # List comprehension...
        new_inds = [price[low:hi].argmax() + low if lab == -2 else
                    price[low:hi].argmin() + low
                    for low, hi, lab in zip(lower, upper, label)]

        # Find maximum within each interval
        new_max = price[new_inds]
        new_global_max = np.max(new_max)

    #print(np.shape(patt_idx))
    #print(np.shape(price))

    #plt.plot(price,label='Original')
    #plt.plot(inv,label='Smoothed')
    #plt.scatter(patt_idx,price[patt_idx],label='peaks')
    #plt.scatter(new_inds,price[new_inds],c='g',label='better peaks')
    #plt.legend()
    #print(len(price),len(inv))
    #plt.plot(np.sqrt(trans.real**2+trans.imag**2))
    #plt.show()

    return peaks_idx, price[peaks_idx]

t0 = time.time()

fft_detect(price[-200:],0.3,method='mine')

t1 = time.time()

fft_detect(price[-200:],0.3,method='first')

t2 = time.time()

fft_detect(price[-200:],0.3,method='second')

t3 = time.time()


print('For Loop: ',t1-t0)
print('First Method: ',t2-t1, round(100*((t1-t0)-(t2-t1))/(t1-t0),2),'% Speedup')
print('Second Method: ',round(t3-t2,5), round(100*((t1-t0)-(t3-t2))/(t1-t0),2),'% Speedup')
print('Third Method: ',)