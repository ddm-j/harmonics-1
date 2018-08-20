import argparse
import time
import subprocess
import warnings
import plotly as py
from plotly import tools
import plotly.graph_objs as go
import datetime
from tqdm import tqdm
import pause
from harmonic_functions import *
#from functionsMaster import *
import multiprocessing
from datetime import timedelta
import os.path

warnings.filterwarnings("ignore",category =RuntimeWarning)

openhour = 1
closehour = 10


#pairs = pd.read_csv('pairs.csv').values[0]

data = pd.read_csv('Data/GBPUSD.csv')
data.columns = ['Date', 'open', 'high', 'low', 'close', 'AskVol']
data = data.set_index(pd.to_datetime(data['Date']))

data = data[['open', 'high', 'low', 'close', 'AskVol']]

data = data.drop_duplicates(keep=False)

data['spread'] = 0.0002

class backtestResults(object):

    def __init__(self,data,custom):

        self.parameters = data[0]
        self.performance = data[1]
        self.trade_info = data[2]
        self.patt_info = data[3]
        self.pairs = data[4]
        self.frame = data[5]
        self.patterns_info = data[6]
        self.custom = custom
        self.local_base = '~/Desktop/harmonics-1/Live\ Testing/BTData/'+self.frame+'/'
        self.server_base = '~/public_html/hedge_vps/Backtests/proto2/'+self.frame+'/'

        filename = [str(i) for i in self.parameters]
        filename = '-'.join(filename)

        self.filename = filename

    def gen_plot(self):

        os.system('mkdir ' + self.local_base + self.filename+'/')

        # Extract Trade Data

        trade_info = self.trade_info.set_index('entry',drop=False)
        trade_dates = self.trade_info.entry
        equity = self.trade_info.equity
        durations = self.trade_info.exit - self.trade_info.entry
        time_difference = trade_dates[1:] - trade_dates[:-1]
        average_freq = sum(time_difference, timedelta())/len(time_difference)

        # Extract Performance Data

        sharpe = self.performance[0]
        apr = self.performance[1]
        acc = self.performance[2]
        exp = self.performance[3]
        mdd,ddd,start_mdd,end_mdd = self.performance[4]

        # Extract Pairwise Data

        total = self.patt_info[0]
        correct = self.patt_info[1]
        pair_pos = np.array([float(i) for i in self.patt_info[2].values[0]])
        pair_neg = np.array([float(i) for i in self.patt_info[3].values[0]])
        pair_acc = 100*pair_pos/(pair_pos+pair_neg)


        pnl_grouped = self.trade_info.groupby('instrument')['pnl'].apply(list)

        pnl_grouped = [pnl_grouped[pair] for pair in pnl_grouped.index.tolist()]

        pnl_pos = [[x for x in pnl_grouped[i] if x > 0] for i in range(0,len(pnl_grouped))]
        pnl_neg = [[abs(x) for x in pnl_grouped[i] if x < 0] for i in range(0,len(pnl_grouped))]

        pnl_mean_pos = [np.mean(x) for x in pnl_pos]
        pnl_mean_neg = [np.mean(x) for x in pnl_neg]

        pair_exp = [10000*((pair_acc[i]/100)*pnl_mean_pos[i] - (1-pair_acc[i]/100.0)*pnl_mean_neg[i]) for i in range(0,len(pnl_mean_pos))]
        pair_exp = [str(round(i,2)) for i in pair_exp]

        pair_acc = [str(round(i)) for i in pair_acc]

        title = 'Multi-Pair Harmonic Pattern Backtest<br>Pairs: '+', '.join(self.pairs)+'<br>'+str(trade_info.entry[0])+' through '+\
                str(trade_info.entry[-1])

        # Plotting

        text_labels = zip(trade_info.instrument,[str(i) for i in trade_dates],[str(i) for i in trade_info.exit],
                          [str(round(i)) for i in trade_info.pos_size],[str(i) for i in durations])

        text_labels = ['Instrument: '+l[0]+'<br>Entry: '+l[1]+'<br>Exit: '+l[2]
                       +'<br>Position Size: '+l[3]+'<br>Trade Duration: '+l[4] for l in text_labels]

        pair_labels = ['<br>'+i[0]+': '+i[1]+'%, '+i[2]+' pips' for i in zip(self.pairs,pair_acc,pair_exp)]
        pair_labels = ''.join(pair_labels)


        trace0 = go.Scatter(x=trade_dates, y=equity,
                            text=text_labels,
                            hoverinfo='text',
                            mode = 'lines+markers',
                            name='Account Equity'
                            )

        trace1 = go.Scatter(x=[start_mdd,end_mdd],
                            y=[trade_info.loc[start_mdd].equity,trade_info.loc[end_mdd].equity],
                            text = ['Draw Down Start','Draw Down End'],
                            hoverinfo='text',
                            mode='markers',
                            name='Draw Down',
                            showlegend=False,
                            marker=dict(
                                color='red'
                            )
                            )

        trace2 = go.Scatter(x=[trade_info.entry.iloc[0]],
                            y=[trade_info.equity.max()],
                            mode='markers',
                            name='Performance Info',
                            text=['Sharpe: '+str(round(sharpe,2))+
                                  '<br>APR: '+str(round(apr,2))+'%'+
                                  '<br>Accuracy: '+str(round(acc,2))+'%'+
                                  '<br>Expectancy: '+str(round(exp,2))+' pips'+
                                  '<br>Maximum Drawdown: '+str(round(100*mdd,2))+'%'+
                                  '<br>Drawdown Duration: '+str(ddd)],
                            hoverinfo='text',
                            marker=dict(
                                color='green',
                                symbol='triangle-left',
                                size=20
                            ))

        trace3 = go.Scatter(x=[trade_info.entry.iloc[0]],
                            y=[trade_info.equity.max()*0.9],
                            mode='markers',
                            name='Pattern Info',
                            text=['Total Patterns Found: '+str(total)+
                                  '<br>Average Downtime: '+str(average_freq)+
                                  '<br>Correct: '+str(correct)+
                                  '<br>-Pairwise Accuracy & Expectancy-'+
                                  pair_labels],
                            hoverinfo='text',
                            marker=dict(
                                color='black',
                                symbol='triangle-right',
                                size=20
                            ))

        layout = go.Layout(xaxis=dict(title='Trades Placed'),
                           yaxis=dict(title='Account Equity ($)'),
                           annotations=[dict(x = start_mdd,
                                             y = trade_info.loc[start_mdd].equity,
                                             ax = 100,
                                             ay = 0,
                                             text = "Start Drawdown",
                                             arrowcolor = "red",
                                             arrowsize = 3,
                                             arrowwidth = 1,
                                             arrowhead = 1),
                                        dict(x=end_mdd,
                                             y=trade_info.loc[end_mdd].equity,
                                             ax = -100,
                                             ay = 0,
                                             text="End Drawdown",
                                             arrowcolor="red",
                                             arrowsize=3,
                                             arrowwidth=1,
                                             arrowhead=1)
                                        ])

        data = [trace0,trace1,trace2,trace3]

        fig = go.Figure(data=data, layout=layout)

        py.offline.plot(fig,filename='BTData/'+self.frame+'/'+self.filename+'/'+self.filename+'.html',auto_open=False)


    def gen_trade_plot(self):

        # Create folder:

        for i in self.patterns_info:

            trace0 = go.Ohlc(x=i['df'].index,
                            open=i['df'].open,
                            high=i['df'].high,
                            low=i['df'].low,
                            close=i['df'].close,
                             name='OHLC Data')
            trace1 = go.Scatter(x=i['pattern_data'].index,
                                y=i['pattern_data'].values,
                                line=dict(color='black'),
                                name='Harmonic Pattern')

            trace2 = go.Scatter(x=[i['pattern_data'].index[-1],i['df'].close.index[-1]],
                                y=[i['df'].close[i['trade_dates'][0]],i['df'].close[i['trade_dates'][0]]],
                                line=dict(
                                    color='green',
                                    width=4,
                                    dash='dash'),
                                name='Entry'
                                )

            trace3 = go.Scatter(x=[i['pattern_data'].index[-1], i['df'].close.index[-1]],
                                y=[i['df'].close[i['trade_dates'][1]], i['df'].close[i['trade_dates'][1]]],
                                line=dict(
                                    color='red',
                                    width=4,
                                    dash='dash'),
                                name='Exit'
                                )

            data = [trace0,trace1,trace2,trace3]

            layout = go.Layout(
                title=i['pattern_info'][0]+' '+i['pattern_info'][1],
                xaxis=dict(
                    rangeslider=dict(
                        visible=False
                    ), title='Date'
                ),
                yaxis=dict(title='Quote Price')
            )

            fig = go.Figure(data=data, layout=layout)
            py.offline.plot(fig,filename='BTData/'+self.frame+'/'+self.filename+'/'+str(i['id'])+'.html',auto_open=False)

    def push2web(self,del_files=True,custom=False):

        # opt_frame is an optional frame to push to the server

        # Send Trade Info to CSV

        selection = ["<input onClick=\'javascript:getVal1();return false;\' type=radio name=\'selection1\' value="
                     +str(i)+ '>' for i in range(0,len(self.patterns_info))]

        self.trade_info['selection'] = selection
        clean_df = self.trade_info[['selection','instrument','entry','exit','pos_size','pnl','equity']]
        clean_df.columns = [['Selection','Pair','Entry','Exit','Postion Size','PnL (pips)','Realized Equity']]
        clean_df['PnL (pips)'] = 10000*clean_df['PnL (pips)']
        clean_df = clean_df.round(2)
        clean_df.to_csv('BTData/'+self.frame+'/'+self.filename+'/'+self.filename+'.csv')

        # Connect Via SSH

        ip,user,passwd = 'hedgefinancial.us', 'hedgefin@146.66.103.215', 'Allmenmustdie1!'
        ext = ['.html','.csv']

        table_fname = self.filename.replace('-','_')

        # Generate Table

        local_file = self.local_base+self.filename+'/'+self.filename
        local_tab = self.local_base+self.filename+'/'+table_fname

        os.system('csvtotable '+local_file+ext[1]+' '+local_tab+ext[0]+' -c \'Backtest Data\' >/dev/null')
        os.system('rm '+local_file+ext[1])

        cmd = 'scp -r -P 18765 %s %s:%s >/dev/null'%(self.local_base+self.filename,user,self.server_base)
        os.system(cmd)

        if del_files:

            os.system('rm -r '+self.local_base+self.filename+' >/dev/null')


class backtestData(object):

    def __init__(self,pairs,frame,n_split,dates=None):
        self.pairs = pairs
        self.frame = frame
        self.dates = dates

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

        if self.dates == None:

            self.historical_hour = hist_data_hour
            self.historical_all = hist_data_all

            self.data_runner = self.historical_hour.iloc[:n_split]
            self.data_feed = self.historical_hour.iloc[n_split:]

        else:
            self.dates = []
            for i in dates:
                nearest = self.nearest([i], hist_data_hour.index)
                self.dates.append(nearest)

            self.historical_hour = hist_data_hour[self.dates[0]:self.dates[1]]
            self.historical_all = hist_data_all

            self.data_runner = self.historical_hour.iloc[:n_split]
            self.data_feed = self.historical_hour.iloc[n_split:]

    def nearest(self, items, pivot):
        return min(items, key=lambda x: abs(x - pivot))

class PatternBot(object):

    def __init__(self,pairs,instrument,data,risk=1,custom=False):

        self.accountID = "101-001-5115623-001"
        self.token = '9632158e473af28669bb91a6fb4e86dd-41aaaa4867de5abf64eda43980f25672' # Insert here
        #self.api = API(access_token=self.token)
        self.instrument = instrument
        self.data = data
        self.perRisk = risk
        self.pipRisk = 20
        self.test = None
        self.train = None
        self.tradeCounter = self.test
        self.tradeTimer = []
        self.predSec = 56
        self.predMin = 59
        self.lastPrediction = 0
        self.state = 1 #something
        self.pairs = pairs
        self.err_allowed = 5.0
        self.custom = custom

    def backtest(self,data_object,params,web_up=True):

        self.frame = data_object.frame

        patterns_info = []

        # Extract Parameters

        peak_param = params[0]
        pattern_err = params[1]
        trade_period = params[2]

        Plot = False

        pair_pos = pd.DataFrame(dict((key,[0]) for key in self.pairs))
        pair_neg = pd.DataFrame(dict((key, [0]) for key in self.pairs))
        pnl = []
        pair_list = []
        quote_list = []
        stop_list = []
        entry_dates = []
        exit_dates = []
        look_ahead = 30
        pip_hist = np.repeat(0.0, look_ahead)
        cl = []
        sizes = []
        patt_cnt = 0
        corr_pats = 0

        # Get data

        self.hist_data = data_object.data_runner

        last_patt_start = 0
        last_trade_time = 0

        # LoopStart

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

                if patterns != None and patterns[-1][0] != last_patt_start and data_object.data_feed.iloc[i].name != last_trade_time:

                    last_patt_start = patterns[-1][0]
                    last_trade_time = data_object.data_feed.iloc[i].name

                    quote_list.append(patterns[-1][-1])

                    patt_cnt += 1

                    # Get Trade Placement Time

                    trade_time = data_object.data_feed.iloc[i].name
                    entry_dates.append(trade_time)

                    # Get Trade Performance

                    sign = -1 if patterns[1] == 'Bearish' else 1
                    if len(data_object.data_feed) > i + trade_period:
                        pips = data_object.data_feed[pair].iloc[i + trade_period] - data_object.data_feed[pair].iloc[i]
                        exit_time = data_object.data_feed.iloc[i + trade_period].name

                    else:
                        pips = data_object.data_feed[pair].iloc[-1] - data_object.data_feed[pair].iloc[i]
                        exit_time = data_object.data_feed.iloc[-1].name

                    if sign == -1 and pips < 0:
                        pips = abs(pips)
                    elif sign == -1 and pips > 0:
                        pips = -pips

                    if len(data_object.data_feed) > i + look_ahead:
                        if sign == 1:
                            pip_hist += data_object.data_feed[pair].iloc[i:i + look_ahead].values - \
                                        data_object.data_feed[pair].iloc[i]
                        elif sign == -1:
                            pip_hist += data_object.data_feed[pair].iloc[i] - data_object.data_feed[pair].iloc[
                                                                              i:i + look_ahead].values

                    # Get indicators

                    #patt_index = data_object.historical_all[pair].index.get_loc(trade_time)

                    #s = get_indicators(data_object.historical_all[pair].iloc[patt_index - 100:patt_index + 1],
                                       #trade_time)

                    #indicators = indicators.append(s.T)

                    if sign == 1:
                        cl.append(0 if pips > 0 else 1)
                    else:
                        cl.append(2 if pips > 0 else 3)

                    exit_dates.append(exit_time)
                    pair_list.append(pair)

                    pnl = np.append(pnl, pips)

                    # Append Pattern Info

                    tmp_patt = self.hist_data[pair][patterns[2]]

                    tmp_dict = {'id':patt_cnt-1,'df':data_object.historical_all[pair][tmp_patt.index[0]:exit_time],
                                'pattern_data':tmp_patt,'pattern_info':[pair,patterns[1]+' '+patterns[0]],
                                'trade_dates':[last_trade_time,exit_time]}
                    patterns_info.append(tmp_dict)

                    if pips > 0:

                        corr_pats += 1
                        pair_pos[pair][0] += 1

                    else:

                        pair_neg[pair][0] += 1

                    if Plot == True:

                        print('Pattern Found ', str(pair),trade_time)

                        idx = patterns[2]
                        start = np.array(idx).min()
                        end = np.array(idx).max()
                        pattern = patterns[3]

                        label = str(pair) + ' ' + patterns[1] + ' ' + patterns[0]

                        print(label)

                        plt.clf()
                        plt.plot(np.arange(start, end + 1), self.hist_data[pair].iloc[-(end-start+1):].values)
                        plt.plot(np.arange(end,end+20),data_object.data_feed[pair].iloc[i:i+20].values)
                        plt.scatter(np.arange(end,end+20),data_object.historical_all[pair].low.iloc[i+500:i+520],c='r')
                        plt.scatter(np.arange(end, end + 20),
                                    data_object.historical_all[pair].high.iloc[i + 500:i + 520], c='g')
                        plt.plot(idx, pattern, c='r')
                        plt.title(label)
                        plt.show()

        risk = self.perRisk
        equity = [1000]

        equity = self.pnl2equity(pnl,sizes,pair_list,quote_list,stop_list,[data_object.historical_all[self.pairs[0]].index.tolist(),entry_dates,exit_dates],equity)

        self.trade_info = pd.DataFrame({'instrument':pair_list,'entry':entry_dates,'exit':exit_dates,'pos_size':sizes,
                                   'pnl':pnl,'equity':equity[1:]})

        start_seconds = data_object.data_feed['EUR_USD'].index[0].toordinal()
        end_seconds = data_object.data_feed['EUR_USD'].index[-1].toordinal()

        self.performance = self.get_performance(self.trade_info,[start_seconds,end_seconds,patt_cnt,corr_pats])

        self.patt_info = [patt_cnt,corr_pats,pair_pos,pair_neg]

        ext_perf = self.performance.copy()

        ext_perf[0:0] = [peak_param,pattern_err]

        self.btRes = backtestResults([[peak_param,pattern_err],
                                      self.performance,self.trade_info,self.patt_info,
                                     self.pairs,self.frame,patterns_info],custom=self.custom)

        if web_up:
            #self.btRes.gen_trade_plot()
            self.btRes.gen_plot()
            #self.btRes.push2web()


        return [peak_param, pattern_err, pip_hist.max(), pip_hist.argmax()]


    def pnl2equity(self,pnl,sizes,pair_list,quote_list,stop_list,dates,equity):

        total_dates = dates[0]
        entry_dates = dates[1]
        exit_dates = dates[2]

        current_eq = equity[0]
        dups = []
        dups_cnt = []

        position_exit = []
        position_profit = []

        for i in range(0, len(total_dates)):

            if total_dates[i] in entry_dates:

                ind = entry_dates.index(total_dates[i])

                print(current_eq)

                cost = current_eq*self.perRisk/100.0

                # Determine Trade Profit

                pnl_i = pnl[ind]
                u_j = pair_list[ind][-3:] == 'JPY'
                flip = pair_list[ind][:3] == 'USD'
                quote = quote_list[ind]

                position = posSizeBT(current_eq, self.perRisk, 20)

                # Update Current Equity

                current_eq = (1.0-self.perRisk/100.0)*current_eq

                sizes.append(position)

                if u_j:

                    print(pnl_i)

                revenue = cost + position * pnl_i

                comission = revenue * (25 / 1000000)

                profit = revenue - comission

                # Check For duplicate:

                if exit_dates[ind] in position_exit:
                    ind2 = position_exit.index(exit_dates[ind])
                    position_profit[ind2] += profit
                    dups[ind2] = True
                    dups_cnt[ind2] += 1
                else:
                    position_profit.append(profit)
                    position_exit.append(exit_dates[ind])
                    dups.append(False)
                    dups_cnt.append(1)

            if total_dates[i] in position_exit:

                ind = position_exit.index(total_dates[i])

                # Update Current Equity

                current_eq += position_profit[ind]

                if dups[ind]:
                    for j in range(dups_cnt[ind]):
                        equity.append(current_eq)

                else:
                    equity.append(current_eq)

                del position_exit[ind]
                del position_profit[ind]
                del dups[ind]
                del dups_cnt[ind]

        if position_profit != []:

            for i in position_profit:

                equity.append(equity[-1]+i)

        return equity

    def get_performance(self,trade_info,extra):

        equity = trade_info.equity.values
        pnl = trade_info.pnl.values
        start_seconds = extra[0]
        end_seconds = extra[1]
        patt_cnt = extra[2]
        corr_pats = extra[3]

        # Percentage returns

        returns = (equity[1:] - equity[:-1]) / equity[:-1]
        returns_df = pd.Series(returns,index=trade_info.entry[1:].values)

        cumreturns = (returns + 1).cumprod() - 1

        # Sharpe Ratio

        sharpe = (returns.mean() / returns.std()) * np.sqrt(len(returns))

        # APR

        t_elapsed = float(end_seconds - start_seconds)

        apr = 100 * (np.exp((float(252) * np.log(equity[-1] / equity[0])) / t_elapsed) - 1)

        # Prediction Accuracy

        acc = 100 * float(corr_pats) / float(patt_cnt)

        # Expectancy

        pos_pnl = pnl[np.where(pnl > 0)[0]]
        neg_pnl = abs(pnl[np.where(pnl < 0)[0]])

        expectancy = 10000 * ((float(len(pos_pnl)) / float(len(pnl))) * np.mean(pos_pnl) - (
                float(len(neg_pnl)) / float(len(pnl))) * np.mean(neg_pnl))

        # Maximum Drawdown

        mdd, start_mdd, end_mdd = self.max_dd(returns_df)

        ddd = end_mdd-start_mdd

        mdd = [mdd,ddd,start_mdd,end_mdd]

        return [sharpe,apr,acc,expectancy,mdd]


    def max_dd(self,returns):
        """Assumes returns is a pandas Series"""
        r = returns.add(1).cumprod()
        dd = r.div(r.cummax()).sub(1)
        mdd = dd.min()
        end = dd.idxmin()
        start = r.loc[:end].idxmax()
        return mdd, start, end

    def walk(self, data, sign, stop=10, u_j=False):

        price = data

        pip_size = 100.0 if u_j else 10000.0

        stop_amount = stop / pip_size
        spread = 2.0/pip_size

        if sign == 1:

            initial_stop_loss = price[0] - stop_amount

            stop_loss = initial_stop_loss

            for i in range(1, len(price)):

                move = price[i] - price[i - 1]

                if move > 0 and (price[i] - stop_amount) > initial_stop_loss:

                    stop_loss = price[i] - stop_amount

                elif price[i] < stop_loss:

                    return (price[i] - spread) - price[0], price.index[i]

            return (price[-1] - spread) - price[0], price.index[-1]


        elif sign == -1:

            initial_stop_loss = price[0] + stop_amount

            stop_loss = initial_stop_loss

            for i in range(1, len(price)):

                move = price[i] - price[i - 1]

                if move < 0 and (price[i] + stop_amount) < initial_stop_loss:

                    stop_loss = price[i] + stop_amount

                elif price[i] > stop_loss:

                    return (price[0] - spread) - price[i], price.index[i]

            return (price[0] - spread) - price[-1], price.index[-1]

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


if __name__ == '__main__':

    # Parse Arguments
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-dates',nargs='+',help='Start and End Date for Backtest, %d-%m-%Y')
    group.add_argument('-frame',help='Frame for test, options: ytd, 1year, 2year, 5year')
    parser.add_argument('-pairs',nargs='+',help='Pairs for backtesting, format XXX_YYY ZZZ_FFF')
    parser.add_argument('-parameters',nargs='+',help='strategy parameters, format stop peak error')
    parser.add_argument('--risk',help='Risk per position, values 1-100')
    args = parser.parse_args()

    if args.dates != None:
        frame = 'Custom'
        dates = [datetime.datetime.strptime(i, '%d-%m-%Y') for i in args.dates]
    else:
        frame = args.frame
        dates = None
    if args.risk == None:
        risk = 1

    pairs = args.pairs
    parameters = args.parameters
    risk = args.risk
    parameters[0] = float(parameters[0])
    parameters[1] = int(parameters[1])
    parameters[2] = float(parameters[2])
    parameters[3] = int(parameters[3])
    risk = float(risk) 


    data = backtestData(n_split=500,pairs=pairs,frame=frame,dates=dates)
    bot = PatternBot(data=data,risk=risk,pairs=pairs,custom=True if frame=='Custom' else False,instrument='test')
    info,params=bot.backtest(data,parameters,web_up=True)
