# Import libraries
import pandas as pd
f = pd.read_csv("L:/Data_Science/Machine_Learning/8) Univariate_Feature_Selection/Students2.csv")

x = f.iloc[:,:-1]
y = f.iloc[:,-1]

# Import various selection transform along with the F_regression mode
from sklearn.feature_selection import SelectKBest, \
                                      SelectPercentile,  \
                                      GenericUnivariateSelect, \
                                      f_regression
                                      
selectork = SelectKBest(score_func=f_regression,k=3)

x_k = selectork.fit_transform(x, y)

# Get F_score nad P_value of selected feature
f_score = selectork.scores_
p_values = selectork.pvalues_

# print the table of feature, F_score and P-values
columns = list(x.columns)

print(" ")
print(" ")
print(" ")

print("   Features  ", "    F_score ", "    P-values" )
print("   -----------","  -----------",' ----------')

for i in range(0,len(columns)):
    f1 = "%4.2f" % f_score[i]
    p1 = "%2.6f" % p_values[i]
    print("    ",columns[i].ljust(12), f1.rjust(8),"  ",p1.rjust(8))
    
# get the column names
cols = selectork.get_support(indices=True)
selected_col= x.columns[cols].tolist()
print(selected_col)

# SelectPercentile
selectorp = SelectPercentile(score_func=f_regression,percentile=50)
x_p = selectorp.fit_transform(x, y)

# Implement Generic Univariate transform
selectorG1 = GenericUnivariateSelect(score_func=f_regression,
                                     mode="k_best",
                                     param=3)
x_g1 = selectorG1.fit_transform(x, y)


# Implement Generic Univariate transform
selectorG2 = GenericUnivariateSelect(score_func=f_regression,
                                     mode="percentile",
                                     param=50)
x_g2 = selectorG2.fit_transform(x, y)

