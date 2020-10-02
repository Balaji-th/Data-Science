#Simple Linear Regression
import pandas as pd

dataset = pd.read_csv("01Students.csv")
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
#calculaten coff. and intercept