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
from random import choices, randint


# Main Strategy Parameters For Tuning

stop_loss = 5 #pips
max_slip_time = 10 #pips
risk_level = 1 # account equity at risk
accn_value = 1000
pattern_error = 20.0 #percentage

filtering_percent = 0.5
sup_res_band = 0.0001

data_resample = False
resampling_frame = 'M'
resampling_frequency = 30

pairs = ['GBPUSD','EURUSD','AUDUSD','NZDUSD']
time_frame = '_ytd'
data = []
prices = []
n = 1000
cut = False

pair_count = [0,0,0,0]

# Load Data

slippage_dist = pd.read_csv('slippage_dist.csv',index_col=0)
slip_val = slippage_dist.index.values


for i in pairs:

    print('Loaded '+i+time_frame+'.csv')

    data1 = pd.read_csv('Data/'+i+time_frame+'.csv')

    if cut:

        data1 = data1.iloc[:n]

    data1.columns = ['Date','open','high','low','close','vol']

    data1.Date = pd.to_datetime(data1.Date,format='%d.%m.%Y %H:%M:%S.%f')

    data1 = data1.set_index(data1.Date)

    data1 = data1[['open','high','low','close','vol']]

    data1 = data1.drop_duplicates(keep=False)

    if data_resample:
        data1 = resampler(data1, timeframe=resampling_frame, length=resampling_frequency)

    price1 = data1.close.copy()

    data.append(data1)
    prices.append(price1)


# Loop Through

pat_length = 5
err_allowed = pattern_error

pats = 0
correct_pats = 0

pnl = []
z_vals = []
strengths = []
trade_dates = []

cnt = 0

for price in prices:

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

                    pats += 1

                    sense = 'Bearish ' if harmonics[j] == -1 else 'Bullish '
                    title = sense + labels[j] + ' Found, '+pairs[cnt]

                    start = current_idx.min()
                    end = current_idx.max()

                    date = data[cnt].iloc[end].name

                    # Determine Slippage Based on Distributions

                    slip_time = randint(0,max_slip_time) # 0 corresponds to 5 seconds slippage
                    slip_prob = slippage_dist.iloc[:,[slip_time]].values

                    slippage = choices(slip_val,slip_prob)

                    trade_dates = np.append(trade_dates, date)
                    pips = walk_forward(price.values[end:], harmonics[j], slippage=slippage[0], stop=stop_loss)
                    pnl = np.append(pnl, pips)


                    #cumpips = pnl.cumsum()

                    if pips > 0:

                        correct_pats += 1
                        pair_count[cnt] += 1

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

    cnt += 1

results = pd.DataFrame({'Date':trade_dates,'pnl':pnl})
results.set_index('Date',inplace=True)
results.sort_index(inplace=True)

trade_dates = results.index
pnl = results.pnl

pos_pnl = pnl[np.where(pnl>0)[0]]
neg_pnl = abs(pnl[np.where(pnl<0)[0]])

expectancy = (float(len(pos_pnl))/float(len(pnl)))*np.mean(pos_pnl) - (float(len(neg_pnl))/float(len(pnl)))*np.mean(neg_pnl)


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


# Printing Info

print('Total Patterns found:',pats)
print('Correct Patterns:',correct_pats)
print('Percentage:',round(100*float(correct_pats)/float(pats)),'%')
print('Expectancy: ',round(expectancy,6)*10000,' Pips Per Trade')
title = 'Account Equity vs. Time, Sharpe: ' + str(round(sharpe,2)) + ', APR '+str(round(apr,2))+'%'

print('-----------------')
print('Pair Success Shares')
for i in range(0,len(pairs)):
    print(pairs[i]+': ',round(100*float(pair_count[i])/float(correct_pats)),'%')

# Plotting

trace0 = go.Scatter(x=trade_dates,y=equity)

layout = go.Layout(title=title,
                   xaxis=dict(title='Trades Placed'),
                   yaxis = dict(title='Account Value ($)')
                   )

data = [trace0]

fig = go.Figure(data=data,layout=layout)

py.offline.plot(fig)