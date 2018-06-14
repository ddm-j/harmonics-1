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
import warnings
import datetime
from tqdm import tqdm
import pause
from harmonic_functions import *
#from functionsMaster import *
import multiprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from datetime import timedelta
import os.path

warnings.filterwarnings("ignore",category =RuntimeWarning)

openhour = 1
closehour = 10


pairs = pd.read_csv('pairs.csv').values[0]

data = pd.read_csv('Data/GBPUSD.csv')
data.columns = ['Date', 'open', 'high', 'low', 'close', 'AskVol']
data = data.set_index(pd.to_datetime(data['Date']))

data = data[['open', 'high', 'low', 'close', 'AskVol']]

data = data.drop_duplicates(keep=False)

data['spread'] = 0.0002


class backtestData(object):

    def __init__(self,frame,n_split):

        self.pairs = pd.read_csv('pairs.csv').values[0]
        pairs = self.pairs
        self.frame = frame

        hist_data_hour = pd.DataFrame()
        hist_data_min = {}
        hist_data_all = {}

        for i in pairs:

            tmp = pd.read_csv(self.frame +'/'+ i + '.csv')
            tmp.columns = ['Date', 'open', 'high', 'low', 'close','volume']

            tmp.Date = pd.to_datetime(tmp.Date, format='%d.%m.%Y %H:%M:%S.%f')

            tmp = tmp.set_index(tmp.Date)

            tmp = tmp[['open', 'high', 'low', 'close']]

            tmp = tmp.drop_duplicates(keep=False)

            hist_data_all.update({i:tmp})

            hist_data_hour[i] = tmp.close

        self.historical_hour = hist_data_hour
        self.historical_all = hist_data_all

       # for i in pairs:

           # tmp = pd.read_csv('1min/' + i + '.csv')
           # tmp.columns = ['Date', 'open', 'high', 'low', 'close','volume']

           # tmp.Date = pd.to_datetime(tmp.Date, format='%d.%m.%Y %H:%M:%S.%f')

            #tmp = tmp.set_index(tmp.Date)

           # tmp = tmp[['open', 'high', 'low', 'close']]

          #  tmp = tmp.drop_duplicates(keep=False)

         #   hist_data_min.update({i:tmp})

        #self.historical_min = hist_data_min

        self.data_runner = self.historical_hour.iloc[:n_split]
        self.data_feed = self.historical_hour.iloc[n_split:]


class PatternBot(object):

    def __init__(self,instrument,data,test=100,train=2000):

        self.accountID = "101-001-5115623-001"
        self.token = '9632158e473af28669bb91a6fb4e86dd-41aaaa4867de5abf64eda43980f25672' # Insert here
        self.api = API(access_token=self.token)
        self.instrument = instrument
        self.data = data
        self.train = train
        self.test = test
        self.perRisk = 1
        self.pipRisk = 20
        self.tradeCounter = self.test
        self.tradeTimer = []
        self.predSec = 56
        self.predMin = 59
        self.lastPrediction = 0
        self.state = 1 #something
        self.pairs = pd.read_csv('pairs.csv').values[0]
        self.err_allowed = 5.0

    def backtest(self,data_object,params):

        # Extract Parameters

        stop_loss = params[0]
        peak_param = params[1]
        pattern_err = params[2]

        Plot = False

        pnl = []
        patt_cnt = 0
        corr_pats = 0

        # Get data

        self.hist_data = data_object.data_runner

        # Begin Backtesting Loop

        for i in range(0,len(data_object.data_feed)):

            # Get New Data and append to historical feed

            self.hist_data = self.hist_data.append(data_object.data_feed.iloc[i])

            # Check for Patterns!

            results_dict = self.loop_check(pattern_err,peak_param)

            if results_dict == None:
                continue

            for j in results_dict:

                pair = j
                patterns = results_dict[j]

                if patterns != None:

                    patt_cnt += 1

                    # Get Trade Placement Time

                    trade_time = data_object.data_feed.iloc[i].name

                    walk_data = data_object.historical_hour[pair][trade_time:]

                    # Walk Forward Through Data

                    sign = -1 if patterns[1] == 'Bearish' else 1
                    pips = self.walk(walk_data,sign,stop=stop_loss)

                    pnl = np.append(pnl,pips)

                    if pips > 0:

                        corr_pats += 1

                    if Plot == True:

                        print('Pattern Found ', str(pair))

                        idx = patterns[2]
                        start = np.array(idx).min()
                        end = np.array(idx).max()
                        pattern = patterns[3]

                        label = str(pair) + ' ' + patterns[1] + ' ' + patterns[0]

                        print(label)

                        plt.clf()
                        plt.plot(np.arange(start, end + 1), self.hist_data[pair].iloc[start:end + 1].values)
                        plt.plot(np.arange(end,end+20),data_object.data_feed[pair].iloc[i:i+20].values)
                        plt.scatter(np.arange(end,end+20),data_object.historical_all[pair].low.iloc[i+100:i+120],c='r')
                        plt.scatter(np.arange(end, end + 20),
                                    data_object.historical_all[pair].high.iloc[i + 100:i + 120], c='g')
                        plt.plot(idx, pattern, c='r')
                        plt.title(label)
                        plt.show()

        risk = self.perRisk
        equity = [1000]

        for i in range(0, len(pnl)):

            position = posSizeBT(equity[i], risk, 20)

            revenue = position * pnl[i]

            comission = revenue * (25 / 1000000)

            profit = revenue - comission

            equity = np.append(equity, equity[i] + profit)

        #plt.plot(equity)
        #plt.show()

        # Percentage returns

        returns = (equity[1:] - equity[:-1]) / equity[:-1]

        cumreturns = (returns + 1).cumprod() - 1

        # Sharpe Ratio

        sharpe = (returns.mean() / returns.std()) * np.sqrt(len(returns))

        # APR

        start_seconds = data_object.data_feed['EUR_USD'].index[0].toordinal()
        end_seconds = data_object.data_feed['EUR_USD'].index[-1].toordinal()

        t_elapsed = float(end_seconds - start_seconds)

        apr = 100 * (np.exp((float(252) * np.log(equity[-1] / equity[0])) / t_elapsed) - 1)

        # Prediction Accuracy

        acc = 100*float(corr_pats) / float(patt_cnt)

        # Expectancy

        pos_pnl = pnl[np.where(pnl > 0)[0]]
        neg_pnl = abs(pnl[np.where(pnl < 0)[0]])

        expectancy = 10000*((float(len(pos_pnl)) / float(len(pnl))) * np.mean(pos_pnl) - (
                    float(len(neg_pnl)) / float(len(pnl))) * np.mean(neg_pnl))

        return [stop_loss,peak_param,pattern_err,sharpe,apr,acc,expectancy]

    def walk(self, data, sign, stop=10):

        price = data
        #low = data.low
        #high = data.high

        stop_amount = float(stop) / float(10000)
        spread = 2.0/10000.0

        if sign == 1:

            initial_stop_loss = price[0] - stop_amount

            stop_loss = initial_stop_loss

            for i in range(1, len(price)):

                move = price[i] - price[i - 1]

                if move > 0 and (price[i] - stop_amount) > initial_stop_loss:

                    stop_loss = price[i] - stop_amount

                elif price[i] < stop_loss:

                    return price[i] - price[0] - spread

            return price[-1] - price[0] - spread


        elif sign == -1:

            initial_stop_loss = price[0] + stop_amount

            stop_loss = initial_stop_loss

            for i in range(1, len(price)):

                move = price[i] - price[i - 1]

                if move < 0 and (price[i] + stop_amount) < initial_stop_loss:

                    stop_loss = price[i] + stop_amount

                elif price[i] > stop_loss:

                    return price[0] - price[i] - spread

            return price[0] - price[-1] - spread

    def read_in_data(self):

        pairs = self.pairs

        hist_data = pd.DataFrame()

        for i in pairs:

            tmp = pd.read_csv('5Min/' + i + '.csv')
            tmp.columns = [['Date', 'open', 'high', 'low', 'close']]

            tmp.Date = pd.to_datetime(tmp.Date, format='%Y.%m.%d %H:%M:%S.%f')

            tmp = tmp.set_index(tmp.Date)

            tmp = tmp[['open', 'high', 'low', 'close']]

            tmp = tmp.drop_duplicates(keep=False)

            hist_data[i] = tmp.close

        self.hist_data = hist_data

    def get_last_hour(self,which=1):

        last_hour = pd.DataFrame()

        for j in range(0,len(self.instrument)):

            params = {}

            params.update({'granularity': 'M5'})
            params.update({'count': 2})
            params.update({'price': 'BA'})
            r = instruments.InstrumentsCandles(instrument=self.instrument[j], params=params)
            rv = self.api.request(r)

            dict = {}

            for i in range(2):
                df = pd.DataFrame.from_dict(rv['candles'][:][i])

                df = df.transpose()

                spread = float(df.loc['ask'].c) - float(df.loc['bid'].c)

                list = df.loc['ask'][['o', 'h', 'l', 'c', 'time']]
                date = df.loc['time'].c
                vol = df.loc['volume'].c

                dict[i] = {'date': date, 'open': float(list.o), 'high': float(list.h), 'low': float(list.l),
                           'close': float(list.c), 'AskVol': float(vol)
                    , 'spread': spread, 'symbol':self.instrument[j]}

            df = pd.DataFrame.from_dict(dict)
            df = df.transpose()

            df = df.set_index(pd.to_datetime(df.date))

            df = df[['open', 'high', 'low', 'close', 'AskVol', 'spread', 'symbol']]

            last_hour = last_hour.append(df.iloc[which])

            last_hour = last_hour[['close','symbol']]


        time = last_hour.index[0]
        values = last_hour.close.values
        symbols = last_hour.symbol.values
        append = pd.DataFrame(data=[values],columns=symbols,index=[time])

        self.hist_data = self.hist_data.append(append)

        self.hist_data.to_csv('Data/Composite_Prices.csv')


    def check_pattern(self,price,err_allowed,range_param):

        pat_length = 5

        peaks_idx, peaks = peak_detect(price.values, peak_range=range_param)

        current_idx = np.array(peaks_idx[-pat_length:])
        current_pat = peaks[-pat_length:]

        XA = current_pat[1] - current_pat[0]
        AB = current_pat[2] - current_pat[1]
        BC = current_pat[3] - current_pat[2]
        CD = current_pat[4] - current_pat[3]

        moves = [XA, AB, BC, CD]

        gar = is_gartley(moves, err_allowed)
        butter = is_butterfly(moves, err_allowed)
        bat = is_bat(moves, err_allowed)
        crab = is_crab(moves, err_allowed)
        shark = is_shark(moves, err_allowed)

        harmonics = np.array([gar, butter, bat, crab])
        labels = ['Gartley', 'Butterfly', 'Bat', 'Crab']

        if np.any(harmonics == 1) or np.any(harmonics == -1):
            for j in range(0, len(harmonics)):
                if harmonics[j] == 1 or harmonics[j] == -1:
                    pattern = labels[j]
                    sense = 'Bearish' if harmonics[j]==-1 else 'Bullish'
                    break
            return [pattern,sense,current_idx,current_pat]

        else:

            return None


    def loop_check(self,patt_err,range):

        results = {}

        for i in self.pairs:

            pattern = self.check_pattern(self.hist_data[i],err_allowed=patt_err,range_param=range)

            if pattern != None:

                results.update({i:pattern})

        if results != {}:

            return results

        else:

            return None



    def position_sizer(self):

        client = self.api

        r = accounts.AccountSummary(self.accountID)

        client.request(r)

        balance = r.response['account']['balance']

        PnL = r.response['account']['unrealizedPL']

        estBalance = float(balance) + float(PnL)

        size = posSize(estBalance, self.perRisk, self.pipRisk, self.recentClose)

        units = size * 1000

        units = round(units)

        return units


    def run(self):

        i = 0

        while 1:

            if i == 0:

                # Get Data

                #self.read_in_data()

                self.hist_data = pd.read_csv('Data/Composite_Prices.csv',index_col=0)

                print('Data Loaded... Beginning Algorithmic Scanning')

            else:

                isWeekday = [True if time.gmtime(time.time()).tm_wday != 5 or 6 else False]
                isTradingHours = [True if 7 < time.gmtime(time.time()).tm_hour < 20 else False]

                now = time.gmtime(time.time())

                if isWeekday and now.tm_min%5 == 0 and now.tm_sec == 30:

                    self.get_last_hour(which=1)

                    print('Checking for Patterns '+str(now.tm_hour)+'-'+str(now.tm_min))

                    patterns,pair = self.loop_check()

                    if patterns != None:

                        print('Pattern Found ',str(pair))

                        idx = patterns[2]
                        start = np.array(idx).min()
                        end = np.array(idx).max()
                        pattern = patterns[3]

                        label = str(pair) + ' ' + patterns[1] + ' ' + patterns[0]

                        print(label)

                        plt.clf()
                        plt.plot(np.arange(start,end+1),self.hist_data[pair].iloc[start:end+1].values)
                        plt.plot(idx,pattern,c='r')
                        plt.title(label)
                        plt.savefig('Patterns/'+str(pair)+str(now.tm_hour)+'-'+str(now.tm_min)+'.png')


                        time.sleep(60*3)
                        #[pattern, sense, current_idx, current_pat]




            i += 1


if False:

    data = backtestData(n_split=500,frame='5year')
    bot = PatternBot(data=data,instrument=pairs)
    bot.backtest(data,[50.0,10,5.0])