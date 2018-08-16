import talib
import pandas as pd
import numpy as np

open = np.random.random(100)
high = np.random.random(100)
low = np.random.random(100)
close = np.random.random(100)

prices = pd.DataFrame(np.array([open,high,low,close]),columns=['open','high','low','close'])

print(prices)