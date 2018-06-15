import pandas as pd
import numpy as np

results = pd.read_csv('OptimizationResults-ytd.csv')

idx = results.acc.idxmax()
print(results.iloc[idx])