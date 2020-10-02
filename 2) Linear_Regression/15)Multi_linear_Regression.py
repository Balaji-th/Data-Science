#Multi Linear Regression
import pandas as pd

dataset = pd.read_csv("02Students.csv")
df = dataset.copy()

#split by column for x and Y
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

#train naad test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =     \
train_test_split(x,y,test_size=0.3,random_state=1234)

#Linear Regression
from sklearn.linear_model import LinearRegression
std_reg = LinearRegression()

#train Regression object on training data
std_reg.fit(x_train,y_train)

#predict the test data
y_predict = std_reg.predict(x_test)

#R-Squared
mlr_score = std_reg.score(x_test,y_test)
#calculaten coff. and intercept
mlr_coeffcient = std_reg.coef_
mlr_intercept = std_reg.intercept_

#equation of the line
#y = 1.31 + 4.67*x1 +5.07*x2

#How mach error we have - RMSE
from sklearn.metrics import mean_squared_error
import math 

mlr_rmse = math.sqrt(mean_squared_error(y_test,y_predict))
