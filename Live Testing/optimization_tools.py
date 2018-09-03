import dispy
import dispy.httpd
from itertools import product
import botProto1
import data
import harmonic_functions
import fast_any_all
from botProto1 import *
import functools
import threading
import time
from datetime import datetime
import logging


class dispy_optimizer(object):

    def __init__(self,frame):

        self.pairs = ['EUR_USD','GBP_USD','AUD_USD','NZD_USD', 'USD_JPY']
        self.error_vals = [5.0,10.0,15.0,20.0,25.0]
        #self.peak_vals = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        self.peak_vals = [5, 10, 15, 20]
        self.trade_periods = [10]
        self.results = pd.DataFrame(columns=['peak', 'error', 'EUR_USD_pips', 'EUR_USD_period',
                                             'GBP_USD_pips', 'GBP_USD_period',
                                             'AUD_USD_pips', 'AUD_USD_period',
                                             'NZD_USD_pips', 'NZD_USD_period',
                                             'USD_JPY_pips', 'USD_JPY_period'])
        self.frame = frame
        training_windows = [1000 , 2500, 5000]
        testing_windows = [500, 750 , 1000, 1500, 2000]

        input_windows = [training_windows, testing_windows]
        windows = np.array(list((product(*input_windows))))

        idx = np.where(windows[:, 0] > windows[:, 1])[0]
        self.windows = windows[idx]

        self.performance_res = pd.DataFrame(columns=['train', 'test', 'sharpe', 'apr', 'acc', 'exp'])

    def prep(self):

        input_data = [self.peak_vals, self.error_vals, self.trade_periods]
        self.grid = list(product(*input_data))
        self.grid = [list(elem) for elem in self.grid]

    def callback(self, job):

        dumb = job()[-2][0]

        if (job.status == dispy.DispyJob.Finished  # most usual case
                or job.status in (dispy.DispyJob.Terminated, dispy.DispyJob.Cancelled,
                                  dispy.DispyJob.Abandoned)):

            self.completed_jobs += 1

            self.now = time.time()

            # Extract Data

            type, window, date_set, total, n, num = job.id

            retval = job()

            p = 100*float(self.completed_jobs) / float(total)

            elapsed = self.now - self.start
            total = elapsed * (1 / (p / 100.0))
            remaining = total - elapsed

            sec_rem = timedelta(seconds=remaining)
            sec_elap = timedelta(seconds=elapsed)

            de = datetime(1, 1, 1) + sec_elap
            dr = datetime(1, 1, 1) + sec_rem

            print(round(p, 2))
            print("Elapsed - %d Days, %d Hours, %d Minutes, %d Seconds" % (de.day - 1, de.hour, de.minute, de.second))
            print("Remaining - %d Days, %d Hours, %d Minutes, %d Seconds" % (dr.day - 1, dr.hour, dr.minute, dr.second))

            self.jobs_cond.acquire()

            if True:#job.id[-1]:

                if type == 'train':

                    self.param_results[repr(window)][n] = self.param_results[repr(window)][n].append(
                        {'peak': retval[0], 'error': retval[1],
                         'EUR_USD_pips': retval[2]['EUR_USD'][0],
                         'EUR_USD_period': retval[2]['EUR_USD'][1],
                         'GBP_USD_pips': retval[2]['GBP_USD'][0],
                         'GBP_USD_period': retval[2]['GBP_USD'][1],
                         'AUD_USD_pips': retval[2]['AUD_USD'][0],
                         'AUD_USD_period': retval[2]['AUD_USD'][1],
                         'NZD_USD_pips': retval[2]['NZD_USD'][0],
                         'NZD_USD_period': retval[2]['NZD_USD'][1],
                         'USD_JPY_pips': retval[2]['USD_JPY'][0],
                         'USD_JPY_period': retval[2]['USD_JPY'][1]
                         }, ignore_index=True)

                elif type == 'test':

                    stitch_info = retval[-1]
                    self.test_results[window][n] = stitch_info

                self.pending_jobs.pop(job.id[-1])

                if len(self.pending_jobs) <= self.lower_bound:
                    self.jobs_cond.notify()

            self.jobs_cond.release()

    def search(self):

        # Training Parameter Storage

        self.param_results = {}
        self.test_params = {}
        self.test_results = {}

        completed = 0

        self.lower_bound, self.upper_bound = 100, 300
        self.jobs_cond = threading.Condition()

        data_ob = data.backtestData(frame=self.frame,pairs=['EUR_USD', 'GBP_USD', 'AUD_USD', 'NZD_USD', 'USD_JPY'],resampled='4H')
        pat_bot = botProto1.PatternBot(pairs=['EUR_USD', 'GBP_USD', 'AUD_USD', 'NZD_USD', 'USD_JPY'],peak_method='scipy')

        cluster = dispy.JobCluster(compute, depends=[harmonic_functions,
                                                     fast_any_all,'Data/GBPUSD.csv'],
                                   setup=functools.partial(setup,data_ob,pat_bot),
                                   callback=self.callback, pulse_interval=5)
        self.pending_jobs = {}
        self.completed_jobs = 0

        master_dates = data_ob.data_feed.index

        shift = 150

        test_date_dict = {}
        train_date_dict = {}

        t = 0

        self.start = time.time()

        for window in self.windows:

            train_dates = [[master_dates[i * window[1]], master_dates[window[0] + i * window[1]]] for
                           i in range(0, int((len(master_dates) - window[0]) / window[1]))]

            test_dates = [
                [master_dates[window[0] + i * window[1] - shift], master_dates[window[0] + (i + 1) * window[1]]] for
                i in range(0, int((len(master_dates) - window[0]) / window[1]))]

            test_date_dict[repr(window)] = test_dates
            train_date_dict[repr(window)] = train_dates

        total = len(self.grid)*sum([len(x) for x in test_date_dict.values()])

        print(total)

        for window in self.windows:

            self.param_results[repr(window)] = {}
            self.test_params[repr(window)] = {}
            self.test_results[repr(window)] = {}
            self.overall_results = pd.DataFrame(columns=['train','test','sharpe','apr','accuracy','expectancy'])

            # Get dates for training and testing:

            #total = sum([len(x) for x in test_date_dict.values()]) * len(self.grid)

            # Submit Training Sets to Nodes

            for n,date_set in enumerate(train_date_dict[repr(window)]):

                self.param_results[repr(window)][n] = pd.DataFrame(columns=['peak', 'error', 'EUR_USD_pips', 'EUR_USD_period',
                                             'GBP_USD_pips', 'GBP_USD_period',
                                             'AUD_USD_pips', 'AUD_USD_period',
                                             'NZD_USD_pips', 'NZD_USD_period',
                                                                            'USD_JPY_pips', 'USD_JPY_period'])
                self.test_params[repr(window)][n] = {}


                for x in self.grid:
                    args = [x, date_set]
                    job = cluster.submit(args)
                    self.jobs_cond.acquire()
                    job.id = ['train',window,date_set,total,n,t]
                    if job.status == dispy.DispyJob.Created or job.status == dispy.DispyJob.Running:
                        self.pending_jobs[t] = job
                        if len(self.pending_jobs) >= self.upper_bound:
                            while len(self.pending_jobs) > self.lower_bound:
                                self.jobs_cond.wait()
                    self.jobs_cond.release()
                    t += 1

        cluster.wait()
        cluster.print_status()

        #cluster.wait()

        # Extract Training Results from Jobs

        print('Beginning Tests')
        t = 0
        self.completed_jobs = 0
        total = sum([len(x) for x in test_date_dict.values()])

        for window_key in self.param_results:

            for n_key in self.param_results[window_key]:

                performance = {}

                for i in self.pairs:

                    pip_lab = i + '_pips'
                    per_lab = i + '_period'

                    pips_idx = self.param_results[window_key][n_key][pip_lab].idxmax()
                    tmp = self.param_results[window_key][n_key][['peak','error',per_lab]]

                    self.test_params[window_key][n_key].update({i:[tmp.iloc[pips_idx].values]})
                    performance.update({i:list(tmp.iloc[pips_idx].values)})

                    # Submit To Cluster
                args = [performance, test_date_dict[window_key][n_key]]
                job = cluster.submit(args)
                self.jobs_cond.acquire()
                job.id = ['test',window_key,test_date_dict[window_key][n_key],total,n_key,t]
                if job.status == dispy.DispyJob.Created or job.status == dispy.DispyJob.Running:
                    self.pending_jobs[t] = job
                    # dispy.logger.info('job "%s" submitted: %s', i, len(self.pending_jobs))
                    if len(self.pending_jobs) >= self.upper_bound:
                        while len(self.pending_jobs) > self.lower_bound:
                            self.jobs_cond.wait()
                self.jobs_cond.release()
                t += 1


        cluster.wait()
        cluster.print_status()

        stitched_results = {}
        equity_master = {}
        trade_info = {}
        perf_master = {}

        cluster.print_status()

        for window_key in self.test_results:

            stitched_results[window_key] = None

            for n_key in sorted(self.test_results[window_key].keys()):

                tmp = self.test_results[window_key][n_key]

                if stitched_results[window_key] == None:
                    stitched_results[window_key] = tmp
                else:
                    for n, item in enumerate(tmp):
                        stitched_results[window_key][n].extend(item)

                # Get Equity Curve for Stitched Result

            equity_master[window_key] = pat_bot.pnl2equity(stitched_results[window_key][0], #pnl
                                                       [], #sizes
                                                       stitched_results[window_key][2], #pair list
                                                       stitched_results[window_key][3], #quote entry
                                                           stitched_results[window_key][4], #quote exit
                                                       stitched_results[window_key][5], # stop list
            [master_dates,stitched_results[window_key][7],stitched_results[window_key][8]],[1000])

            trade_info[window_key] = pd.DataFrame(
                {'instrument': stitched_results[window_key][2],
                 'entry': stitched_results[window_key][7],
                 'exit': stitched_results[window_key][8],
                 'pos_size': stitched_results[window_key][1],
                 'pnl': stitched_results[window_key][0],
                 'equity': equity_master[window_key][1:]})

            perf_master[window_key] = pat_bot.get_performance(trade_info[window_key],
                                                          [stitched_results[window_key][9][0],
                                                           stitched_results[window_key][10][-1],
                                                           sum(stitched_results[window_key][11]),
                                                           sum(stitched_results[window_key][12])])[0:4]

        lens = []

        for key in perf_master:

            l = key.replace('array([','')
            l = l.replace('])','')
            l = [int(s) for s in l.split(',')]

            self.overall_results = self.overall_results.append({
                'train':l[0],'test':l[1],'sharpe':perf_master[key][0],
                'apr':perf_master[key][1],'accuracy':perf_master[key][2],
                'expectancy':perf_master[key][3]
            }, ignore_index=True)

        equity_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in equity_master.items() ]))

        cluster.close()

        return self.overall_results, equity_df


def compute(args):
    #bot = args[0]
    bot = pattern_bot
    data = data_object
    parameters = args[0]
    dates = args[1]
    retval = bot.backtest(data,parameters,dates=dates,web_up=False)

    return retval


def setup(data_ob, pat_bot):

    global data_object, pattern_bot

    data_object = data_ob
    pattern_bot = pat_bot

    return 0

def cb(status, node, job):

    if status == dispy.DispyJob.Finished:

        print(round(100*float(job.id[-1])/float(900),2),'%')