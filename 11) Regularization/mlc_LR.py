# Effect of Lasso and and Ridfe on multicollinearity and 
# Coefficient of linear Regression

# Import Pandas for Data Processing
import pandas as pd
dataset = pd.read_csv("E:/Data_Science/Machine_Learning/11) Regularization/mcl.csv")
df = dataset.copy()

# Split data into x and y
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

correlation = x.corr()

# import all the Regression
from sklearn.linear_model import Lasso, Ridge, LinearRegression
lr = LinearRegression()
lr.fit(x, y)
lr_coeff = lr.coef_
lr_intercept = lr.intercept_

# Lasso Regression 
lasso = Lasso(alpha=10)
lasso.fit(x, y)
lasso_coeff = lasso.coef_
lasso_intercept = lasso.intercept_

# Ridge Regression
ridge = Ridge(alpha=100)
ridge.fit(x, y)
ridge_coeff = ridge.coef_
ridge_intercept = ridge.intercept_



