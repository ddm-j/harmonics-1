import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import choices


# The purpose of this script is to estimate the slippage distribution of a trade

# Import our historical data

data = pd.read_csv('Data/EURUSD_slippage.csv')

#data = data.iloc[:10000]

data.columns = ['Date','open','high','low','close','vol']

data.Date = pd.to_datetime(data.Date,format='%d.%m.%Y %H:%M:%S.%f')

data = data.set_index(data.Date)

data = data[['open','high','low','close','vol']]

data = data.drop_duplicates(keep=False)

price = data.close.copy()

# Latency

slippage_probs = pd.DataFrame(index = np.arange(-10.0,10.0,0.1))
bins = 50

slippage = []

for k in range(5,30):

    for i in range(k,len(price)):

        tmp = 10000*(price.iloc[i]-price.iloc[i-k])
        tmp = round(tmp,1)
        slippage.append(tmp)

    v, counts = np.unique(slippage,return_counts=True)
    p = counts/sum(counts)

    tmp_frame = pd.DataFrame(index=v,data=p,columns=[[k]])

    slippage_probs = slippage_probs.join(tmp_frame,how='outer')

slippage_probs = slippage_probs.fillna(value=0.0)

print(slippage_probs.columns)
slippage_probs.to_csv('slippage_dist.csv')



