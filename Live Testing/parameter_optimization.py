#!/software/development/Build/Anaconda3-4.4.0/envs/python-3.6/bin/python

#SBATCH --job-name=FOREX Optimization Backtest
#SBATCH --output=res.txt

#

#SBATCH --ntasks=81
#SBATCH --time=10:00
#SBATCH --nodes=3
#SBATCH --mem-per-cpu=100

import os
import sys
print(sys.version)
sys.path.append(os.getcwd())

from itertools import repeat
from itertools import product
from botProto1 import *
import warnings
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("-frame")
args = parser.parse_args()

warnings.filterwarnings("ignore",category =RuntimeWarning)


class optimizer(object):

    def __init__(self,n_proc,frame):

        self.n_proc = n_proc
        self.error_vals = [2.0,5.0,10.0]#,15.0,20.0]
        self.stop_vals = [0.25,0.5,0.75]#,1.0,1.5,2.0]
        self.peak_vals = [5,10,15]#,20]
        self.atrs = [5, 7, 10]#, 14, 21]
        self.results = pd.DataFrame(columns=['stop','peak','error','atr_range','sharpe','apr','acc','exp'])
        self.frame = frame

    def prep(self):
        self.data = backtestData(frame=self.frame,n_split=500,pairs=['EUR_USD','GBP_USD','AUD_USD','NZD_USD'])
        self.bot = PatternBot(data=data,instrument=pairs,pairs=['EUR_USD','GBP_USD','AUD_USD','NZD_USD'])
        #parameters = {'stop':self.stop_vals,'peak':self.peak_vals,'error':self.error_vals,'atrs':self.atrs}
        #self.grid = ParameterGrid(parameters)
        input_data = [self.stop_vals,self.peak_vals,self.error_vals,self.atrs]
        self.grid = list(product(*input_data))
        self.grid = [list(elem) for elem in self.grid]

        print(len(self.grid))


    def ret_func(self,retval):

        retval = retval[1]

        now = time.time()
        self.results = self.results.append({'stop':retval[0],'peak':retval[1],'error':retval[2],'atr_range':retval[3],'sharpe':retval[4],
                                            'apr':retval[5],'acc':retval[6],'exp':retval[7]},ignore_index=True)

        percent = 100*float(len(self.results))/float(len(self.grid))

        elapsed = now - self.start
        total = elapsed*(1/(percent/100.0))
        remaining = total - elapsed

        print(round(percent),'% ','[Sharpe APR ACC EXP] = [',round(self.results.sharpe.max(),2),round(self.results.apr.max(),2),
              round(self.results.acc.max(),2),round(self.results.exp.max(),2),']')
        print('Elapsed: ',round(elapsed),'Remaining: ',round(remaining))

        if round(percent)%5==0:
            self.results.to_csv('OptimizationResults-'+self.frame+'.csv')

    def search(self):

        self.start = time.time()

        p = multiprocessing.Pool(processes=self.n_proc)
        print(self.n_proc)
        results = []

        for x, y in zip(repeat(self.data),self.grid):

            r = p.apply_async(self.bot.backtest,(x,y),callback=self.ret_func)

        p.close()
        p.join()

if __name__ == '__main__':

    #multiprocessing.freeze_support()

    opt = optimizer(n_proc=81,frame=args.frame)
    opt.prep()
    print('Data Prepped, beginning search')
    start_time = time.time()
    opt.search()
    end_time = time.time()
    print(end_time-start_time,'seconds elapsed')
