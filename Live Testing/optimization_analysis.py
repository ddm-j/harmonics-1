import pandas as pd
import numpy as np

results = pd.read_csv('OptimizationResults-1year.csv')

idx = results.sharpe.idxmax()
print(results.iloc[idx])