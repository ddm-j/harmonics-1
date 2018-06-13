import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from harmonic_patterns import *
from scipy.signal import argrelextrema
from tqdm import tqdm
import plotly as py
from plotly import tools
import plotly.graph_objs as go
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from datetime import timedelta


# Main Strategy Parameters For Tuning

stop_loss = 5 #pips
slippage = 4 #pips
risk_level = 1 # account equity at risk
accn_value = 1000
pattern_error = 20.0 #percentage

filtering_percent = 0.5
sup_res_band = 0.0001

data_resample = False
resampling_frame = 'M'
resampling_frequency = 30


# Import our historical data

data = pd.read_csv('Data/EURUSD.csv')

#data = data.iloc[:10000]

data.columns = ['Date','open','high','low','close','vol']

data.Date = pd.to_datetime(data.Date,format='%d.%m.%Y %H:%M:%S.%f')

data = data.set_index(data.Date)

data = data[['open','high','low','close','vol']]

data = data.drop_duplicates(keep=False)

price = data.close.copy()

if data_resample:

    price = resampler(price,timeframe=resampling_frame,length=resampling_frequency)

print('Data Resampled')


# Loop Through

pat_length = 5
err_allowed = pattern_error

pats = 0
correct_pats = 0

pnl = []
z_vals = []
strengths = []
trade_dates = []

#plt.ion()

for i in tqdm(range(200,len(price.values))):

    peaks_idx,peaks = peak_detect(price.values[:i],peak_range=5)

    current_idx = np.array(peaks_idx[-pat_length:])
    current_pat = peaks[-pat_length:]

    XA = current_pat[1] - current_pat[0]
    AB = current_pat[2] - current_pat[1]
    BC = current_pat[3] - current_pat[2]
    CD = current_pat[4] - current_pat[3]

    moves = [XA,AB,BC,CD]

    gar = is_gartley(moves,err_allowed)
    butter = is_butterfly(moves,err_allowed)
    bat = is_bat(moves,err_allowed)
    crab = is_crab(moves,err_allowed)
    #shark = is_shark(moves,err_allowed)

    harmonics = np.array([gar,butter,bat,crab])
    labels = ['Gartley','Butterfly','Bat','Crab']

    if np.any(harmonics == 1) or np.any(harmonics == -1):


        for j in range(0,len(harmonics)):

            if harmonics[j] == 1 or harmonics[j] == -1:

                # Check Support and Resistance?

                if False:

                    means,stds,revs = sup_res_levels(price.values[i-500:i],p=filtering_percent,delta=sup_res_band)
                    nearest_level_idx = find_nearest(means,price.values[i])
                    z_val = (price.values[i] - means[nearest_level_idx])/stds[nearest_level_idx]

                    z_vals = np.append(z_vals,z_val)
                    strengths = np.append(strengths,revs[nearest_level_idx])


                pats += 1

                sense = 'Bearish ' if harmonics[j] == -1 else 'Bullish '
                title = sense + labels[j] + ' Found'

                start = current_idx.min()
                end = current_idx.max()
                date = data.iloc[end].name

                trade_dates = np.append(trade_dates,date)

                pips = walk_forward(price.values[end:],harmonics[j],slippage=slippage,stop=stop_loss)

                pnl = np.append(pnl,pips)

                #cumpips = pnl.cumsum()

                if pips > 0:

                    correct_pats += 1

                #plt.clf()
                #lbl = 'Accuracy ' + str(100*float(correct_pats)/float(pats)) + ' %'
                #plt.plot(cumpips,label=lbl)
                #plt.legend()
                #plt.pause(0.05)

                if False:
                    plt.title(title)
                    plt.plot(np.arange(start,end+15),price.values[start:end+15])
                    plt.plot(current_idx,current_pat)
                    plt.scatter(current_idx,current_pat)
                    plt.show()

                break

time_difference = trade_dates[1:]-trade_dates[:-1]
average = sum(time_difference, timedelta())/len(time_difference)

print('Average Frequency of Patterns', average)


risk = risk_level
equity = [accn_value]

for i in range(0,len(pnl)):

    position = posSize(equity[i],risk,20)

    revenue = position*pnl[i]

    comission = revenue*(25/1000000)

    profit = revenue - comission

    equity = np.append(equity,equity[i]+profit)

#resultsCSV = pd.DataFrame(equity[1:],index=trade_dates)

#resultsCSV.to_csv('Results/B.csv')


# Percentage returns

returns = (equity[1:] - equity[:-1])/equity[:-1]

cumreturns = (returns+1).cumprod() - 1

# Sharpe Ratio

sharpe = (returns.mean()/returns.std())*np.sqrt(len(returns))

# APR

start_seconds = price.index[0].toordinal()
end_seconds = price.index[-1].toordinal()

t_elapsed = float(end_seconds-start_seconds)


apr = 100*(np.exp((float(252)*np.log(equity[-1]/equity[0]))/t_elapsed) - 1)

# Prediction Accuracy

acc = float(correct_pats)/float(pats)


# Z Value and Strength Data Analysis

if False:

    z_vals = np.array(z_vals)
    strengths = np.array(strengths)
    pnl = np.array(pnl)

    inds = strengths.argsort()

    sorted_z = z_vals[inds]
    sorted_strs = strengths[inds]
    sorted_pnl = pnl[inds]

    plt.scatter(sorted_strs,sorted_pnl)
    plt.title('Level Confidence vs. Gains')
    plt.xlabel('Support/Resistance Strength')
    plt.ylabel('Gains (Pips)')
    plt.show()


# Printing Info

print('Total Patterns found:',pats)
print('Correct Patterns:',correct_pats)
print('Percentage:',round(100*float(correct_pats)/float(pats)),'%')
title = 'Account Equity vs. Time, Sharpe: ' + str(sharpe) + ', APR '+str(apr)+'%'

# Plotting

trace0 = go.Scatter(x=trade_dates,y=equity)

layout = go.Layout(title=title,
                   xaxis=dict(title='Trades Placed'),
                   yaxis = dict(title='Account Value ($)')
                   )

data = [trace0]

fig = go.Figure(data=data,layout=layout)

py.offline.plot(fig)