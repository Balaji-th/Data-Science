# Import Pandas as pd
import pandas as pd
dataset = pd.read_csv("E:/Data_Science/Machine_Learning/11) Regularization/ridge.csv")
df = dataset.copy()

# Split into x and y
x = df.iloc[:,:-1] 
y = df.iloc[:,-1]

from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

x_plt = [0,1,2,3,4]
lasso_1 = [0,0.5,1,2,4]

for i,l in enumerate(lasso_1):
    lasso = Lasso(alpha=l)
    lasso.fit(x,y)
    
    lasso_coeff = lasso.coef_
    lasso_intercept = lasso.intercept_
    y_plt = lasso.predict(pd.DataFrame(x_plt))
    
    plt.figure(1)
    plt.plot(x_plt,y_plt)
    plt.ylim(ymin=0,ymax=9)
    plt.xlim(xmin=0,xmax=6)
    plt.text(x_plt[-1],y_plt[-1],
             " y="+str('%.2f' %lasso_coeff)+
             " * x" +
             " + " +
             str("%.2f" %lasso_intercept)+
             "     for \u03BB or \u03B1 = "+str(l),fontsize=12)