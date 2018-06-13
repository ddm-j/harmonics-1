import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from harmonic_functions import *
from tqdm import tqdm

# Import our historical data

data = pd.read_csv('Data/EURUSD.csv')

data.columns = [['Date','open','high','low','close','vol']]

data.Date = pd.to_datetime(data.Date,format='%d.%m.%Y %H:%M:%S.%f')

data = data.set_index(data.Date)

data = data[['open','high','low','close','vol']]

data = data.drop_duplicates(keep=False)

price = data.close.copy()

# Find Peaks

err_allowed = 20.0/100

#plt.ion()

pnl = []
trade_dates = []
correct_pats = 0
pats = 0

plt.ion()

for i in tqdm(range(100,len(price.values))):

    current_idx,current_pat,start,end = peak_detect(price.values[:i],order=5)

    XA = current_pat[1] - current_pat[0]
    AB = current_pat[2] - current_pat[1]
    BC = current_pat[3] - current_pat[2]
    CD = current_pat[4] - current_pat[3]

    moves = [XA,AB,BC,CD]

    gart = is_gartley(moves,err_allowed)
    butt = is_butterfly(moves, err_allowed)
    bat = is_bat(moves, err_allowed)
    crab = is_crab(moves, err_allowed)

    harmonics = np.array([gart, butt, bat, crab])
    labels = ['Gartley', 'Butterfly', 'Bat', 'Crab']

    if np.any(harmonics == 1) or np.any(harmonics == -1):

        pats += 1

        for j in range(0,len(harmonics)):

            if harmonics[j] == 1 or harmonics[j] == -1:

                sense = 'Bearish ' if harmonics[j]==-1 else 'Bullish '
                label = sense + labels[j] + ' Found'

                start = np.array(current_idx).min()
                end = np.array(current_idx).max()
                date = data.iloc[end].name
                trade_dates = np.append(trade_dates,date)


                pips = walk_forward(price.values[end:],harmonics[j],slippage=4,stop=5)

                pnl = np.append(pnl,pips)

                cumpips = pnl.cumsum()

                if pips>0:

                    correct_pats +=1

                lbl = 'Accuracy ' + str(100*float(correct_pats)/float(pats)) + ' %'

                plt.clf()
                plt.plot(cumpips,label=lbl)
                plt.legend()
                plt.pause(0.05)


                #plt.title(label)
                #plt.plot(np.arange(start,i+15),price.values[start:i+15])
                #plt.plot(current_idx,current_pat,c='r')
                #plt.show()



