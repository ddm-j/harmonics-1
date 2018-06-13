import pandas as pd
import numpy as np

results = pd.read_csv('OptimizationResults.csv')

for i in results.columns[1:]:

    results[i] = results[i].str.strip('[]').astype(float)

idx = results.sharpe.idxmax()
print(results.iloc[idx])