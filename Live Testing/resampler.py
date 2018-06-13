import pandas as pd
import numpy as np


pairs = pd.read_csv('pairs.csv')

hist_data = pd.DataFrame()

for i in pairs:

    tmp = pd.read_csv('Minutes/' + i + '.csv')
    tmp.columns = [['Date', 'open', 'high', 'low', 'close', 'vol']]

    tmp.Date = pd.to_datetime(tmp.Date, format='%d.%m.%Y %H:%M:%S.%f')

    tmp = tmp.set_index(tmp.Date)

    tmp = tmp[['open', 'high', 'low', 'close', 'vol']]

    tmp = tmp.drop_duplicates(keep=False)

    p_tmp = tmp.close.copy()

    p_tmp = p_tmp.resample('5T').ohlc()

    p_tmp.to_csv('5Min/'+i+'.csv')
