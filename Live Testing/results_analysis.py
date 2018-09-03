import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

key = repr(np.array([10000, 750]))

df = pd.read_csv('equity_results_scipy.csv')


for i in df.columns:

    plt.plot(df[i])

plt.show()

