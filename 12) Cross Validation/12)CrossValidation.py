#-------------------------------------------------------
# Compare multi classification using Cross Validation
#-------------------------------------------------------

# Import libraries
import pandas as pd
data = pd.read_csv('E:/Data_Science/Machine_Learning/12) Cross Validation/04+-+decisiontreeAdultIncome.csv')

# Create Dummy variables
data_prep = pd.get_dummies(data,drop_first=True)

# Create x and y variables
x = data_prep.iloc[:,:-1]
y = data_prep.iloc[:,-1]

# Import and train DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=1234)

# Import and train RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=1234)

# Import and train SupportVectorClassifier
from sklearn.svm import SVC
svc = SVC(kernel='rbf',gamma=0.5)

# Import and train SupportVectorClassifier
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

# Preform Cross Validation
from sklearn.model_selection import cross_validate
cv_result_dtc = cross_validate(dtc, x, y, cv=10, return_train_score=True)
cv_result_rfc = cross_validate(rfc, x, y, cv=10, return_train_score=True)
cv_result_svc = cross_validate(svc, x, y, cv=10, return_train_score=True)
cv_result_lr = cross_validate(lr, x, y, cv=10, return_train_score=True)

# Get Average Score for models
import numpy as np
dtc_test_average = np.average(cv_result_dtc['test_score'])
rfc_test_average = np.average(cv_result_rfc['test_score'])
svc_test_average = np.average(cv_result_svc['test_score'])
lr_test_average = np.average(cv_result_lr['test_score'])

dtc_train_average = np.average(cv_result_dtc['train_score'])
rfc_train_average = np.average(cv_result_rfc['train_score'])
svc_train_average = np.average(cv_result_svc['train_score'])
lr_train_average = np.average(cv_result_lr['train_score'])

# print the results 
print()
print()
print('        ','Decision Tree  ', 'Random Forest  ','Support Vector ','LogisticRegression')
print('        ','---------------', '---------------','---------------','------------------')

print('Test  : ',
      round(dtc_test_average, 4), '        ',
      round(rfc_test_average, 4), '        ',
      round(svc_test_average, 4), '        ',
      round(lr_test_average,4),'          ')

print('Train : ',
      round(dtc_train_average, 4), '        ',
      round(rfc_train_average, 4), '        ',
      round(svc_train_average, 4),'        ',
      round(lr_train_average,4))







