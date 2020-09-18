#------------------------------------------------
#  F-test in univarient Feature Selection
#------------------------------------------------

# Import Libraries
import pandas as pd

# Read the file
f = pd.read_csv("L:/Data_Science/Machine_Learning/8) Univariate_Feature_Selection/Students2.csv")

# Split the column into Dependent(Y) and independent(x) Feature
x = f.iloc[:,:-1]
y = f.iloc[:,-1]

# Linear Regression using original data
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = \
    train_test_split(x, y,test_size=0.4,random_state=1234)

lr.fit(x_train, y_train)

y_predict = lr.predict(x_test)

from sklearn.metrics import mean_squared_error
import math

rmse = math.sqrt(mean_squared_error(y_test, y_predict))

# F_regression from sklearn
from sklearn.feature_selection import f_regression as fr
result_fr = fr(x,y)

f_score = result_fr[0]
p_values = result_fr[1]

# print the table of feature, F_score and P-values
columns = list(x.columns)

print(" ")
print(" ")
print(" ")

print(" Features  ",  "F_score  ", "P-values" )
print("-----------","-----------",'----------')

for i in range(0,len(columns)):
    f1 = "%4.2f" % f_score[i]
    p1 = "%2.6f" % p_values[i]
    print("    ",columns[i].ljust(12), f1.rjust(8),"  ",p1.rjust(8))
    
# Compare the model performance with Selected
x_train_n = x_train[["Hours","sHours"]]
x_test_n = x_test[["Hours","sHours"]]

lr1 = LinearRegression()

lr1.fit(x_train_n, y_train)

y_predict_n = lr1.predict(x_test_n)

rmse_n = math.sqrt(mean_squared_error(y_test, y_predict_n))
    
    
    
    
    
    
    
