#!/software/development/Build/Anaconda3-4.4.0/envs/python-3.6/bin/python -u
#SBATCH --output=dr.txt

#

#SBATCH --ntasks=2
#SBATCH --time=10:00
#SBATCH --nodes=2
#SBATCH --mem-per-cpu=8G


# MAIN SCRIPT SECTION
# Scipt imports

import os
import sys
import subprocess
print(sys.version)
sys.path.append(os.getcwd())

from itertools import repeat
from itertools import product
import pandas as pd
import numpy as np
import botProto1
from botProto1 import *
import harmonic_functions
import warnings
import argparse
import time
import dispy
import pickle

# Initiate Dispy Node Server on Compute Nodes

nodes = subprocess.check_output('echo $SLURM_JOB_NODELIST',shell=True)
print(nodes)
nodes = nodes.decode('utf-8')
nodes = nodes[8:-2]
nodes = [int(nodes[0]),int(nodes[-1])]

print(nodes)

#for i in nodes:

        #os.system('srun dispynode.py --clean --daemon & &>/dev/null')

os.system('srun dispynode.py --clean --daemon & &>/dev/null')

# Command Line Arguments

parser = argparse.ArgumentParser()
parser.add_argument("-frame")
args = parser.parse_args()

warnings.filterwarnings("ignore",category =RuntimeWarning)

# Script Class

class optimizer(object):

    def __init__(self,n_proc,frame):

        self.n_proc = n_proc
        self.error_vals = [2.0,5.0,10.0,15.0,20.0,25.0,30.0,35.0]
        self.peak_vals = [5,10,15,20,25,30,35]
        self.trade_periods = [10]
        self.results = pd.DataFrame(columns=['peak','error','cum_pips','period'])
        self.frame = frame

    def prep(self):
        #parameters = {'stop':self.stop_vals,'peak':self.peak_vals,'error':self.error_vals,'atrs':self.atrs}
        #self.grid = ParameterGrid(parameters)
        input_data = [self.peak_vals,self.error_vals,self.trade_periods]
        self.grid = list(product(*input_data))
        self.grid = [list(elem) for elem in self.grid]


    def ret_func(self,job):

        retval = job.result

        retval = retval[0:7]

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

        data = botProto1.backtestData(frame=self.frame, n_split=500,
                                           pairs=['EUR_USD', 'GBP_USD', 'AUD_USD', 'NZD_USD'])
        bot = botProto1.PatternBot(data=[0], instrument=[0], pairs=['EUR_USD', 'GBP_USD', 'AUD_USD', 'NZD_USD'])

        cluster = dispy.JobCluster(compute,depends=[backtestResults,backtestData,PatternBot,harmonic_functions,
                                                    'Data/GBPUSD.csv',botProto1])
        print('Cluster Created')
        jobs = []
        id = 0
        for x in self.grid:
            args = [bot,data,x]
            job = cluster.submit(args)
            job.id = id
            jobs.append(job)
            id += 1

        print('Backtests Beginning')
        cluster.print_status()

        for job in jobs:
            retval = job()
            print(job.exception)

            self.results = self.results.append(
                {'peak': retval[0], 'error': retval[1], 'cum_pips':retval[2], 'period':retval[3]}, ignore_index=True)

            p = 100*float(len(self.results))/float(len(self.grid))

            print(p,'% Completion')

        idx = self.results.cum_pips.idxmax()
        print(self.results.iloc[idx])



def compute(args):
    bot = args[0]
    data = args[1]
    parameters = args[2]
    retval = bot.backtest(data,parameters,web_up=False)

    return retval

if __name__ == '__main__':

    #multiprocessing.freeze_support()

    opt = optimizer(n_proc=81,frame=args.frame)
    opt.prep()
    print('Data Prepped, beginning search')
    start_time = time.time()
    opt.search()
    end_time = time.time()
    print(end_time-start_time,'seconds elapsed')

