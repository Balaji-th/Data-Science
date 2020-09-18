#------------------------------------------------------------
# Implement Recursive Feature Elimination
# Predict Product purchace for the Bank Telemarketing dataset
#------------------------------------------------------------

# Import Libraries
import pandas as pd 

# Read the dataset
f = pd.read_csv("L:/Data_Science/Machine_Learning/9) Recursive_Feature_Elimination/bank.csv")

# drop the duration column because we want predict weather customer
# buy our product or not before call.hence call duration is not important
f = f.drop("duration",axis=1)

# Split column into X and Y
x = f.iloc[:,:-1]
y = f.iloc[:,-1]

# create dummy variables for categorical features
x = pd.get_dummies(x,drop_first=True)
y = pd.get_dummies(y,drop_first=True)

# Split dataset Train and test
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = \
    train_test_split(x, y,test_size=0.3,random_state=1234,stratify=y)

# import Random forest classifier
from sklearn.ensemble import RandomForestClassifier

# Default Random Forest object
rfc1 = RandomForestClassifier(random_state=1234)
rfc1.fit(x_train, y_train)
y_predict = rfc1.predict(x_test)

# Score and Evaluate the model
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test,y_predict)
score1 = rfc1.score(x_test, y_test)

# import RFE
from sklearn.feature_selection import RFE
rfc2 = RandomForestClassifier(random_state=1234)
rfe = RFE(estimator=rfc2, n_features_to_select=30,step=1)

rfe.fit(x,y)

x_train_rfe = rfe.transform(x_train)
x_test_rfe = rfe.transform(x_test)

# create a new model using reduced features
# Random Forest object using Recursive feature elumination
rfc2.fit(x_train_rfe, y_train)
y_predict = rfc2.predict(x_test_rfe)

# Score and Evaluate the model
from sklearn.metrics import confusion_matrix
cm_rfe = confusion_matrix(y_test,y_predict)
score_rfe = rfc2.score(x_test_rfe, y_test)

# get Column names
columns = list(x.columns)

# Get the ranking of the features.Ranking 1 for feature
ranking = rfe.ranking_

# Get the feature importance score
feature_importances = rfc1.feature_importances_

# create the dataframe of the Features selection
# Ranking and their importance
rfe_selected = pd.DataFrame()

rfe_selected = pd.concat([pd.DataFrame(columns),
                         pd.DataFrame(ranking),
                         pd.DataFrame(feature_importances)],axis=1)

rfe_selected.columns = ["Feature_importances","ranking","columns"]





