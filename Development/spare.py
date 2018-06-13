data2 = pd.read_csv('Data/EURUSD.csv')

#data = data.iloc[:10000]

data2.columns = ['Date','open','high','low','close','vol']

data2.Date = pd.to_datetime(data2.Date,format='%d.%m.%Y %H:%M:%S.%f')

data2 = data2.set_index(data2.Date)

data2 = data2[['open','high','low','close','vol']]

data2 = data2.drop_duplicates(keep=False)

price2 = data2.close.copy()