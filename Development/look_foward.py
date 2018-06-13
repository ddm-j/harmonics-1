import pandas as pd


A = pd.read_csv('A.csv',index_col=0)
B = pd.read_csv('B.csv',index_col=0)

A = A.values
B = B.values

d = len(A) - len(B)

A = A[:-d]


print(A-B)