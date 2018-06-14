import pandas as pd
import numpy as np

results = pd.read_csv('OptimizationResults.csv')

idx = results.sharpe.idxmax()
print(results.iloc[idx])