import numpy as np
import pandas as pd
import plotly as py
import plotly.graph_objs as go
from datetime import timedelta
import os

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

    def __init__(self,pairs,frame,resampled=False,dates=None):
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

            if resampled:

                tmp = tmp.resample(resampled).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})

            hist_data_all.update({i:tmp})

            hist_data_hour[i] = tmp.close


        if self.dates == None:

            self.data_feed = hist_data_hour
            self.historical_all = hist_data_all

        else:
            self.dates = []
            for i in dates:
                nearest = self.nearest([i], hist_data_hour.index)
                self.dates.append(nearest)

            self.data_feed = hist_data_hour[self.dates[0]:self.dates[1]]
            self.historical_all = hist_data_all

        check_dates = [pd.to_datetime('01.01.2017 05:00:00.00', format='%d.%m.%Y %H:%M:%S.%f'),
                       pd.to_datetime('01.01.2017 23:00:00.00', format='%d.%m.%Y %H:%M:%S.%f')]

        self.data_feed = self.data_feed.fillna(method='ffill')

    def nearest(self, items, pivot):
        return min(items, key=lambda x: abs(x - pivot))
