import dispy
from itertools import product
import botProto1
import harmonic_functions
from botProto1 import *


class dispy_optimizer(object):

    def __init__(self,frame):

        self.pairs = ['EUR_USD','GBP_USD','AUD_USD','NZD_USD']
        self.error_vals = [5.0,10.0,15.0,20.0,25.0]
        self.peak_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.trade_periods = [10]
        self.results = pd.DataFrame(columns=['peak', 'error', 'EUR_USD_pips', 'EUR_USD_period',
                                             'GBP_USD_pips', 'GBP_USD_period',
                                             'AUD_USD_pips', 'AUD_USD_period',
                                             'NZD_USD_pips', 'NZD_USD_period'])
        self.frame = frame
        training_windows = [1000]  # , 2500, 5000, 7500, 10000]
        testing_windows = [500, 750]  # , 1000, 1500, 2000]

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
        self.param_test = {}

        data = botProto1.backtestData(frame=self.frame,pairs=['EUR_USD', 'GBP_USD', 'AUD_USD', 'NZD_USD'])
        bot = botProto1.PatternBot(pairs=['EUR_USD', 'GBP_USD', 'AUD_USD', 'NZD_USD'],peak_method='fft')

        cluster = dispy.JobCluster(compute, depends=[backtestResults, backtestData, PatternBot, harmonic_functions,
                                                     'Data/GBPUSD.csv', botProto1])
        jobs = []

        master_dates = data.data_feed.index

        shift = 200

        test_date_list = []

        for window in self.windows:

            self.param_results[repr(window)] = {}
            self.param_test[repr(window)] = {}

            # Get dates for training and testing:

            train_dates = [[master_dates[i*window[1]],master_dates[window[0]+i*window[1]]] for
                           i in range(0,int((len(master_dates)-window[0])/window[1]))]

            test_dates = [[master_dates[window[0]+i*window[1]-shift],master_dates[window[0]+(i+1)*window[1]]] for
                i in range(0,int((len(master_dates)-window[0])/window[1]))]

            test_date_list.append(test_dates)

            # Submit Training Sets to Nodes

            for n,date_set in enumerate(train_dates):

                self.param_results[repr(window)][n] = pd.DataFrame(columns=['peak', 'error', 'EUR_USD_pips', 'EUR_USD_period',
                                             'GBP_USD_pips', 'GBP_USD_period',
                                             'AUD_USD_pips', 'AUD_USD_period',
                                             'NZD_USD_pips', 'NZD_USD_period'])
                self.param_test[repr(window)][n] = {}

                for x in self.grid:
                    args = [bot, data, x, date_set]
                    job = cluster.submit(args)
                    job.id = ['train',window,date_set,n]
                    jobs.append(job)

        # Extract Training Results from Jobs

        for job in jobs:

            # Get Job ID For Identification

            dummy, window, date_set, n = job.id

            retval = job()

            if job.exception != None:
                print(job.exception)

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

                    tmp = self.param_results[window_key][n_key][['peak','error',per_lab]]

                    self.param_test[window_key][n_key].update({i:[tmp.iloc[pips_idx].values]})
                    performance.update({i:[tmp.iloc[pips_idx].values]})

                # Submit To Cluster

                args = [bot, data, performance, test_dates[n_key]]
                job = cluster.submit(args)
                job.id = ['test',window_key,test_dates[n_key],n_key]
                jobs.append(job)

        return performance


def compute(args):
    bot = args[0]
    data = args[1]
    parameters = args[2]
    retval = bot.backtest(data,parameters,web_up=False)

    return retval