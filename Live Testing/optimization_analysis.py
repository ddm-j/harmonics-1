import pandas as pd
import numpy as np

results = pd.read_csv('OptimizationResults-1year.csv')

idx = results.exp.idxmax()
print(results.iloc[idx])