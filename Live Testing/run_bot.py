import argparse
from botProto1 import *
from data import *
import time

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

    pairs = ['EUR_USD', 'GBP_USD', 'AUD_USD', 'NZD_USD'] if args.pairs == ['all'] else args.pairs
    parameters = args.parameters
    risk = args.risk
    parameters[0] = int(parameters[0])
    parameters[1] = float(parameters[1])
    parameters[2] = int(parameters[2])
    risk = float(risk)


    data = backtestData(n_split=10,pairs=pairs,frame=frame,dates=dates)
    bot = PatternBot(pairs=pairs,risk=risk,custom=True if frame=='Custom' else False)

    t0 = time.time()
    x1, x2, x3, x4=bot.backtest(data,parameters,web_up=False)
    t1 = time.time()

    print(t1-t0)