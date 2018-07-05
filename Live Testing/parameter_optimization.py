from itertools import repeat
from sklearn.model_selection import ParameterGrid
from botProto1 import *
import warnings
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("frame")
args = parser.parse_args()

warnings.filterwarnings("ignore",category =RuntimeWarning)

os.system("taskset -p 0xff %d" % os.getpid())


class optimizer(object):

    def __init__(self,n_proc,frame):

        self.n_proc = n_proc
        self.error_vals = [2.0,5.0,10.0,15.0,20.0]
        self.stop_vals = [0.25,0.5,0.75,1.0,1.5,2.0]
        self.peak_vals = [5,10,15,20]
        self.atrs = [5, 7, 10, 14, 21]
        self.results = pd.DataFrame(columns=['stop','peak','error','atr_range','sharpe','apr','acc','exp'])
        self.frame = frame

    def prep(self):

        self.data = backtestData(frame=self.frame,n_split=500,pairs=['EUR_USD','GBP_USD','AUD_USD','NZD_USD'])
        self.bot = PatternBot(data=data,instrument=pairs,pairs=['EUR_USD','GBP_USD','AUD_USD','NZD_USD'])
        parameters = {'stop':self.stop_vals,'peak':self.peak_vals,'error':self.error_vals,'atrs':self.atrs}
        self.grid = ParameterGrid(parameters)

        stops = [d['stop'] for d in self.grid]
        peaks = [d['peak'] for d in self.grid]
        error = [d['error'] for d in self.grid]
        atrs = [d['atrs'] for d in self.grid]

        self.grid = list(zip(stops,peaks,error,atrs))


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

        # Push Results to the Web

        # Create HTML Code

        selection = ["<input onClick=\'javascript:getVal();return false;\' type=radio name=selection value="
                     +str(i[0])+'-'+str(round(i[1]))+'-'+str(i[2])+'-'+str(round(i[3]))+'>'  for i in zip(self.results.stop,
                                                                             self.results.peak,
                                                                             self.results.error,self.results.atr_range)]
        self.results['selection'] = selection

        ip, user, passwd = 'hedgefinancial.us', 'hedgefin@146.66.103.215', 'Allmenmustdie1!'
        self.results = self.results[['selection','sharpe','apr','acc','exp','stop','atr_range','peak','error']]
        self.results.columns = [['Selection','Sharpe Ratio','APR','Accuracy','Expectancy (pips)','ATR Factor','ATR Range','Peak Parameter','Error']]
        self.results = self.results.round(2)


        self.results.to_csv('BTData/'+self.frame+'/master.csv')

        filepath = '~/public_html/hedge_vps/Backtests/proto2/' + self.frame + '/'
        additional_path = '~/Desktop/harmonics-1/Live\ Testing/BTData/'+self.frame+'/master.html'

        os.system('csvtotable '+ 'BTData/'+self.frame+'/master.csv BTData/'+self.frame+'/master.html -c \'Available Backtests\' >/dev/null')

        cmd = 'scp -P 18765 %s %s:%s' % (additional_path, user, filepath)
        os.system(cmd)
        os.system('rm '+additional_path+' '+additional_path.replace('.html','.csv'))

        print('***************************')
        print('Exiting, Optimization Complete')

if __name__ == '__main__':

    #multiprocessing.freeze_support()

    opt = optimizer(n_proc=4,frame=args.frame)
    opt.prep()
    print('Data Prepped, beginning search')
    opt.search()
