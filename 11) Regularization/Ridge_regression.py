# Import Pandas
import pandas as pd 

# Read the CSV File
dataset = pd.read_csv("L:/Data_Science/Machine_Learning/11) Regularization/ridge.csv")
df = dataset.copy()

# Split into X and Y 
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Import Regid Regression
from sklearn.linear_model import Ridge
l=10
ridge = Ridge(alpha=l)
ridge.fit(x, y)

# Coefficient and Intercept
coeff = ridge.coef_
Intercept = ridge.intercept_

# OLS : Y=2X
# Ridge : Y= 0.8333 + 1.667X

x_plt = [0,1,2,3,4]
y_plt = ridge.predict(pd.DataFrame(x_plt))

import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(x_plt,y_plt)
plt.ylim(ymin=0,ymax=9)
plt.xlim(xmin=0,xmax=6)
# Y = mx + b
plt.text(x_plt[-1],y_plt[-1],
         " Y = "         +
         str('%.2f' %coeff)  +
         "* X + "         +
         str("%.2f" %Intercept) +
         "  for  \u03BB or \u03B1 = "   +
         str(l),
         fontsize=12
         )
# Impact of Varies Alpha/Lamda on Ridge Regression