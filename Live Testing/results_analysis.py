import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

key = repr(np.array([7500, 2000]))

df = pd.read_csv('equity_results_scipy.csv')

for i in df.columns:

    plt.plot(df[i].dropna())

plt.show()

