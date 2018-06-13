import json
from oandapyV20 import API
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.instruments as instruments
from oandapyV20.contrib.requests import MarketOrderRequest
from oandapyV20.exceptions import V20Error
import logging
import pandas as pd
import numpy as np
import harmonic_functions
import time
import datetime
import pause
from harmonic_functions import *
from functionsMaster import *
import multiprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from datetime import timedelta
import os.path



pairs = pd.read_csv('pairs.csv').values[0]


data = pd.DataFrame()


for i in pairs:

    tmp = pd.read_csv('Data/'+i+'.csv')
    tmp.columns = [['Date', 'open', 'high', 'low', 'close', 'AskVol']]

    tmp.Date = pd.to_datetime(tmp.Date, format='%d.%m.%Y %H:%M:%S.%f')

    tmp = tmp.set_index(tmp.Date)

    tmp = tmp.drop_duplicates(keep=False)

    data[i] = tmp.close


print(data.head())



#data = pd.read_csv('Data/GBPUSD.csv')
#data.columns = [['Date', 'open', 'high', 'low', 'close', 'AskVol']]
#data = data.set_index(pd.to_datetime(data['Date']))
#
#data = data[['open', 'high', 'low', 'close', 'AskVol']]
#
#data = data.drop_duplicates(keep=False)
#
#data['spread'] = 0.0002