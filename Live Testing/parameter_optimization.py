
import logging
import pandas as pd
import numpy as np
import harmonic_functions
import time
import datetime
from tqdm import tqdm
from itertools import repeat

import pause
from harmonic_functions import *
import multiprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from datetime import timedelta
import os.path
import random
from sklearn.model_selection import ParameterGrid
from botProto1 import *
import warnings

warnings.filterwarnings("ignore",category =RuntimeWarning)

class optimizer(object):

    def __init__(self,n_proc,frame):

        self.n_proc = n_proc
        self.error_vals = [2.0,5.0,10.0,15.0,20.0,30.0]
        self.stop_vals = [5.0,10.0,15.0,20.0,25.0,30.0,40.0,50.0,60.0]
        self.peak_vals = [5,10,15,20]
        self.results = pd.DataFrame(columns=['stop','peak','error','sharpe','apr','acc','exp'])
        self.frame = frame

    def prep(self):

        self.data = backtestData(frame=self.frame,n_split=500)
        self.bot = PatternBot(data=data,instrument=pairs)
        parameters = {'stop':self.stop_vals,'peak':self.peak_vals,'error':self.error_vals}
        self.grid = ParameterGrid(parameters)
        self.grid = [list(d.values()) for d in self.grid]


    def ret_func(self,retval):


        self.results = self.results.append({'stop':retval[0],'peak':retval[1],'error':retval[2],'sharpe':retval[3],
                                            'apr':retval[4],'acc':retval[5],'exp':retval[6]},ignore_index=True)
        percent = 100*float(len(self.results))/float(len(self.grid))

        print(round(percent,2),'% - [Sharp, APR, ACC, EXP] = [',round(self.results.sharpe.max(),2),
              round(self.results.apr.max(),2),round(self.results.acc.max(),2),round(self.results.exp.max(),2),']')


    def search(self):


        p = multiprocessing.Pool(processes=self.n_proc)

        for x, y in zip(repeat(self.data),self.grid):
            p.apply_async(self.bot.backtest,(x,y),callback=self.ret_func)

        p.close()
        p.join()

        self.results.to_csv('OptimizationResults.csv')

        print('***************************')
        print('Exiting, Optimization Complete')

if __name__ == '__main__':

    multiprocessing.freeze_support()

    opt = optimizer(n_proc=4,frame='ytd')
    opt.prep()
    print('Data Prepped, beginning search')
    opt.search()
