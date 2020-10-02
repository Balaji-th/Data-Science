import pandas as pd
import matplotlib.pyplot as plt

f = pd.read_csv("03-corr.csv")

f["t0"] = pd.to_numeric(f["t0"],downcast="float") #cov to float

plt.acorr(f["t0"],maxlags=10) #k=10

# create timelag dataset
t_1 = f['t0'].shift(+1).to_frame()

t_2 = f['t0'].shift(+2).to_frame()


