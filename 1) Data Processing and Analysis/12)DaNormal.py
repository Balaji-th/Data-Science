import pandas as pd
dataset=pd.read_csv('loan_small.csv')
cleandata = dataset.dropna()
data_to_scale = cleandata.iloc[:, 2:5]

#Standard_Normalization
from sklearn.preprocessing import StandardScaler
scale_=StandardScaler()
ss_scale = scale_.fit_transform(data_to_scale)

#MinMax_Normalization
from sklearn.preprocessing import minmax_scale
mm_scaler = minmax_scale(data_to_scale)
