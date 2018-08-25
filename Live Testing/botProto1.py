import argparse
import warnings
import datetime
from tqdm import tqdm
from harmonic_functions import *
from data import *
import cProfile, pstats, io
import fast_any_all as faa

warnings.filterwarnings("ignore",category =RuntimeWarning)

def profile(fnc):

    def inner(*args, **kwargs):

        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sortby)
        ps.print_stats(0.3)
        print(s.getvalue())

        return retval

    return inner



class PatternBot(object):

    def __init__(self,pairs,risk=1,custom=False):

        self.perRisk = risk
        self.pipRisk = 20
        self.test = None
        self.train = None
        self.pairs = pairs
        self.custom = custom


    @profile
    def backtest(self,data_object,params,web_up=True):

        self.frame = data_object.frame

        patterns_info = []

        # Extract Parameters

        params_dict = {'EUR_USD':params,'GBP_USD':params,'AUD_USD':params,'NZD_USD':params}
        params_dict['EUR_USD'][-1] = 19
        params_dict['GBP_USD'][-1] = 26
        params_dict['NZD_USD'][-1] = 29
        params_dict['AUD_USD'][-1] = 9

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
        empty_lists = np.repeat(0.0,look_ahead)
        pip_hist = {'EUR_USD':empty_lists.copy(),'GBP_USD':empty_lists.copy(),'NZD_USD':empty_lists.copy(),
                    'AUD_USD': empty_lists.copy()}
        cl = []
        sizes = []
        patt_cnt = 0
        corr_pats = 0

        # Get data

        self.hist_data = data_object.data_runner

        last_patt_start = 0
        last_trade_time = 0

        # LoopStart

        for i in tqdm(range(0,len(data_object.data_feed))):


            # Get New Data and append to historical feed

            self.hist_data = self.hist_data.append(data_object.data_feed.iloc[i])

            # Check for Patterns!

            results_dict = self.loop_check(params_dict)

            if results_dict == None:
                continue

            for j in results_dict:
                pair = j
                patterns = results_dict[j]

                # Determine Parameters Based on Pair

                peak_param = params_dict[pair][0]
                pattern_err = params_dict[pair][1]
                trade_period = params_dict[pair][2]


                if patterns[-1][0] != last_patt_start and data_object.data_feed.iloc[i].name != last_trade_time:

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

                    if len(data_object.data_feed) > i + look_ahead:
                        if sign == 1:
                            pip_hist[pair] += data_object.data_feed[pair].iloc[i:i + look_ahead].values - \
                                        data_object.data_feed[pair].iloc[i]

                        elif sign == -1:
                            pip_hist[pair] += data_object.data_feed[pair].iloc[i] - data_object.data_feed[pair].iloc[
                                                                              i:i + look_ahead].values

                    # Get indicators

                    #patt_index = data_object.historical_all[pair].index.get_loc(trade_time)

                    #s = get_indicators(data_object.historical_all[pair].iloc[patt_index - 100:patt_index + 1],
                                       #trade_time)

                    #indicators = indicators.append(s.T)

                    #if sign == 1:
                    #    cl.append(0 if pips > 0 else 1)
                    #else:
                    #    cl.append(2 if pips > 0 else 3)

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

                    # Do you want to Plot each pattern?

                    self.plot_pattern(data_object,pair,patterns,trade_time,i) if False else 0

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

        pip_res = {'EUR_USD': [pip_hist['EUR_USD'].max(), pip_hist['EUR_USD'].argmax()],
                   'GBP_USD': [pip_hist['GBP_USD'].max(), pip_hist['GBP_USD'].argmax()],
                   'AUD_USD': [pip_hist['AUD_USD'].max(), pip_hist['AUD_USD'].argmax()],
                   'NZD_USD': [pip_hist['NZD_USD'].max(), pip_hist['NZD_USD'].argmax()]}


        return [peak_param, pattern_err, pip_res, self.performance]

    def plot_pattern(self,data_object,pair, patterns, trade_time, i):

        print('Pattern Found ', str(pair), trade_time)

        idx = patterns[2]
        start = np.array(idx).min()
        end = np.array(idx).max()
        pattern = patterns[3]

        label = str(pair) + ' ' + patterns[1] + ' ' + patterns[0]

        print(label)

        plt.clf()
        plt.plot(np.arange(start, end + 1), self.hist_data[pair].iloc[-(end - start + 1):].values)
        plt.plot(np.arange(end, end + 20), data_object.data_feed[pair].iloc[i:i + 20].values)
        plt.scatter(np.arange(end, end + 20), data_object.historical_all[pair].low.iloc[i + 500:i + 520], c='r')
        plt.scatter(np.arange(end, end + 20),
                    data_object.historical_all[pair].high.iloc[i + 500:i + 520], c='g')
        plt.plot(idx, pattern, c='r')
        plt.title(label)
        plt.show()


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

        try:
            mdd = dd.min()
            end = dd.idxmin()
            start = r.loc[:end].idxmax()
        except:
            mdd = 0
            start = 0
            end = 0
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


    def check_pattern(self,price,err_allowed,range_param):

        pat_length = 5

        peaks_idx, peaks = peak_detect(price.values, peak_range=range_param)

        current_idx = np.array(peaks_idx[-pat_length:])
        current_pat = peaks[-pat_length:]

        if len(current_idx) < 5:

            return None

        moves = [current_pat[1] - current_pat[0], #XA
                 current_pat[2] - current_pat[1], #AB
                 current_pat[3] - current_pat[2], #BC
                 current_pat[4] - current_pat[3]] #CD

        if (moves[0] > 0 and moves[1] < 0 and moves[2] > 0 and moves[3] < 0) or (moves[0] < 0 and moves[1] > 0 and moves[2] < 0 and moves[3] > 0):

            pass

        else:

            return None

        harmonics = np.array([is_gartley(moves, err_allowed),
                              is_butterfly(moves, err_allowed),
                              is_bat(moves, err_allowed),
                              is_crab(moves, err_allowed)])

        labels = ['Gartley', 'Butterfly', 'Bat', 'Crab']

        if faa.any(abs(harmonics) == 1):

            idx = np.where(abs(harmonics) == 1.)[0][0]
            pattern = labels[idx]
            sense = 'Bearish' if harmonics[idx]==-1 else 'Bullish'

            return [pattern,sense,current_idx,current_pat]

        else:

            return None


    def loop_check(self,params):

        results = {}

        for i in self.pairs:

            patt_err = params[i][1]
            range = params[i][0]

            if len(self.hist_data) < 201:

                return None

            pattern = self.check_pattern(self.hist_data[i].iloc[-200:],err_allowed=patt_err,range_param=range)


            if pattern != None:

                results.update({i:pattern})

        if results != {}:

            return results

        else:

            return None
