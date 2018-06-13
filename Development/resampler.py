import pandas as pd
import numpy as np

# Import our historical data

data = pd.read_csv('Data/EURUSDmins.csv')

data.columns = [['Date','open','high','low','close','vol']]

data.Date = pd.to_datetime(data.Date,format='%d.%m.%Y %H:%M:%S.%f')

data = data.set_index(data.Date)

data = data[['open','high','low','close','vol']]

data = data.drop_duplicates(keep=False)

price = data.close.copy()

price = price.resample('30T').ohlc()

print(price)

price.to_csv('Data/EURUSD_30min.csv')