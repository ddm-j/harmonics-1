import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx


# Import data

data = pd.read_csv('Data/EURUSD.csv')

#data = data.iloc[:50000]

data.columns = ['Date','open','high','low','close','vol']

data.Date = pd.to_datetime(data.Date,format='%d.%m.%Y %H:%M:%S.%f')

data = data.set_index(data.Date)

data = data[['open','high','low','close','vol']]

data = data.drop_duplicates(keep=False)

price = data.close.copy()


# Test FFT

n = 500

x = np.arange(len(data[0:n]+1))
y = data.close[0:n]

rft = np.fft.rfft(y)

real = rft.real
imag = rft.imag
mag = np.sqrt(real**2 + imag**2)

p = 0.5

rft[round(p*len(rft)):] = 0

y_smooth = np.fft.irfft(rft)

dy = np.gradient(y_smooth)

zero_index = np.where(np.diff(np.sign(dy)))[0] + 1

peak_values = y_smooth[zero_index]

def get_blocks(values,delta):
    mi, ma = 0, 0
    result = []
    temp = []
    for v in sorted(values):
        if not temp:
            mi = ma = v
            temp.append(v)
        else:
            if abs(v - mi) < delta and abs(v - ma) < delta:
                temp.append(v)
                if v < mi:
                    mi = v
                elif v > ma:
                    ma = v
            else:
                if len(temp) > 1:
                    result.append(temp)
                mi = ma = v
                temp = [v]
    return result

blocks = get_blocks(peak_values,0.005)

means = []
strengths = []

for i in blocks:

    means = np.append(np.mean(i),means)
    strengths = np.append(len(i),strengths)

strengths = (strengths - strengths.min())/(strengths.max()-strengths.min())

#print(redacted_blocks)

if True:

    plt.plot(x, y, label='Original')
    plt.plot(x, y_smooth, label='Smoothed')
    plt.hlines(means,x.min(),x.max(),colors=cmx.cool(strengths))
    plt.scatter(zero_index,y_smooth[zero_index],label="zero derivative")
    plt.legend(loc=0).draggable()
    plt.show()