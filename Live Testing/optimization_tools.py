import dispy
import dispy.httpd
from itertools import product
import botProto1
import data
import harmonic_functions
import fast_any_all
from botProto1 import *
import functools
import logging


class dispy_optimizer(object):

    def __init__(self,frame):

        self.pairs = ['EUR_USD','GBP_USD','AUD_USD','NZD_USD']
        self.error_vals = [15.0,20.0,25.0]#[5.0,10.0,15.0,20.0,25.0]
        self.peak_vals = [0.2, 0.3, 0.4]#, 0.5, 0.6, 0.7]
        self.trade_periods = [10]
        self.results = pd.DataFrame(columns=['peak', 'error', 'EUR_USD_pips', 'EUR_USD_period',
                                             'GBP_USD_pips', 'GBP_USD_period',
                                             'AUD_USD_pips', 'AUD_USD_period',
                                             'NZD_USD_pips', 'NZD_USD_period'])
        self.frame = frame
        training_windows = [1000]# , 2500]#, 5000, 7500, 10000]
        testing_windows = [500, 750]# , 1000,]# 1500, 2000]

        input_windows = [training_windows, testing_windows]
        windows = np.array(list((product(*input_windows))))

        idx = np.where(windows[:, 0] > windows[:, 1])[0]
        self.windows = windows[idx]

        self.performance_res = pd.DataFrame(columns=['train', 'test', 'sharpe', 'apr', 'acc', 'exp'])

    def prep(self):

        input_data = [self.peak_vals, self.error_vals, self.trade_periods]
        self.grid = list(product(*input_data))
        self.grid = [list(elem) for elem in self.grid]

    def search(self):

        # Training Parameter Storage

        self.param_results = {}
        self.test_params = {}
        self.test_results = {}

        completed = 0

        data_ob = data.backtestData(frame=self.frame,pairs=['EUR_USD', 'GBP_USD', 'AUD_USD', 'NZD_USD'])
        pat_bot = botProto1.PatternBot(pairs=['EUR_USD', 'GBP_USD', 'AUD_USD', 'NZD_USD'],peak_method='fft')

        cluster = dispy.JobCluster(compute, depends=[harmonic_functions,
                                                     fast_any_all,'Data/GBPUSD.csv'],
                                   setup=functools.partial(setup,data_ob,pat_bot),
                                   pulse_interval=1000, cluster_status=cb, reentrant=True)
        jobs = []

        master_dates = data_ob.data_feed.index

        shift = 200

        test_date_dict = {}

        t = 0

        for window in self.windows:

            self.param_results[repr(window)] = {}
            self.test_params[repr(window)] = {}
            self.test_results[repr(window)] = {}
            self.overall_results = pd.DataFrame(columns=['train','test','sharpe','apr','accuracy','expectancy'])

            # Get dates for training and testing:

            train_dates = [[master_dates[i*window[1]],master_dates[window[0]+i*window[1]]] for
                           i in range(0,int((len(master_dates)-window[0])/window[1]))]

            test_dates = [[master_dates[window[0]+i*window[1]-shift],master_dates[window[0]+(i+1)*window[1]]] for
                i in range(0,int((len(master_dates)-window[0])/window[1]))]

            test_date_dict[repr(window)] = test_dates

            # Submit Training Sets to Nodes

            for n,date_set in enumerate(train_dates):

                self.param_results[repr(window)][n] = pd.DataFrame(columns=['peak', 'error', 'EUR_USD_pips', 'EUR_USD_period',
                                             'GBP_USD_pips', 'GBP_USD_period',
                                             'AUD_USD_pips', 'AUD_USD_period',
                                             'NZD_USD_pips', 'NZD_USD_period'])
                self.test_params[repr(window)][n] = {}


                for x in self.grid:

                    args = [x, date_set]
                    job = cluster.submit(args)
                    job.id = ['train',window,date_set,n,t]
                    jobs.append(job)
                    t += 1

        #cluster.wait()

        total = sum([len(x) for x in test_date_dict.values()])*len(self.grid)

        print(total)

        cluster.print_status()

        # Extract Training Results from Jobs


        for job in jobs:

            if job.exception != None:
                print(job.exception)

            # Get Job ID For Identification

            dummy, window, date_set, n, num = job.id

            retval = job()

            self.param_results[repr(window)][n] = self.param_results[repr(window)][n].append(
                {'peak': retval[0], 'error': retval[1],
                 'EUR_USD_pips': retval[2]['EUR_USD'][0],
                 'EUR_USD_period': retval[2]['EUR_USD'][1],
                 'GBP_USD_pips': retval[2]['GBP_USD'][0],
                 'GBP_USD_period': retval[2]['GBP_USD'][1],
                 'AUD_USD_pips': retval[2]['AUD_USD'][0],
                 'AUD_USD_period': retval[2]['AUD_USD'][1],
                 'NZD_USD_pips': retval[2]['NZD_USD'][0],
                 'NZD_USD_period': retval[2]['NZD_USD'][1]}, ignore_index=True)

        # Get Best Performance & Submit to Cluster


        jobs = []
        for window_key in self.param_results:

            for n_key in self.param_results[window_key]:

                performance = {}

                for i in self.pairs:

                    pip_lab = i + '_pips'
                    per_lab = i + '_period'

                    pips_idx = self.param_results[window_key][n_key][pip_lab].idxmax()

                    print(self.param_results[window_key][n_key][pip_lab])
                    print(pips_idx)

                    tmp = self.param_results[window_key][n_key][['peak','error',per_lab]]

                    self.test_params[window_key][n_key].update({i:[tmp.iloc[pips_idx].values]})
                    performance.update({i:list(tmp.iloc[pips_idx].values)})

                    # Submit To Cluster
                args = [performance, test_date_dict[window_key][n_key]]
                job = cluster.submit(args)
                job.id = ['test',window_key,test_date_dict[window_key][n_key],n_key]
                jobs.append(job)

        cluster.print_status()

        # Get Performance of Tests

        for job in jobs:

            # Get Job ID For Identification

            dummy, window, date_set, n = job.id

            retval = job()

            if job.exception != None:
                print(job.exception)

            stitch_info = retval[-1]
            self.test_results[window][n] = stitch_info

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

            equity_master[window_key] = bot.pnl2equity(stitched_results[window_key][0],
                                                       [],
                                                       stitched_results[window_key][2],
                                                       stitched_results[window_key][3],
                                                       stitched_results[window_key][4],
            [master_dates,stitched_results[window_key][6],stitched_results[window_key][7]],[1000])

            trade_info[window_key] = pd.DataFrame(
                {'instrument': stitched_results[window_key][2],
                 'entry': stitched_results[window_key][6],
                 'exit': stitched_results[window_key][7],
                 'pos_size': stitched_results[window_key][1],
                 'pnl': stitched_results[window_key][0],
                 'equity': equity_master[window_key][1:]})

            perf_master[window_key] = bot.get_performance(trade_info[window_key],
                                                          [stitched_results[window_key][8][0],
                                                           stitched_results[window_key][9][-1],
                                                           sum(stitched_results[window_key][10]),
                                                           sum(stitched_results[window_key][11])])[0:4]

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