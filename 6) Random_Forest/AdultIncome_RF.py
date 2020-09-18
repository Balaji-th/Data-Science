#---------------------------------------------------
# Random_Forest of Ensemble_learning of Decision_Tree
# Predict income of adult based on census data
#-----------------------------------------------------

# Import Libraries
import pandas as pd

# Read and copy Dataset
data = pd.read_csv("L:/Data_Science/Machine_Learning/6) Random_Forest/04 - decisiontreeAdultIncome.csv")

# Check for Null values
data.isnull().sum(axis=0)
# Check data type
data.dtypes     

# create dummy variables
data_prep = pd.get_dummies(data,drop_first=True)

# Create X-independent and Y-dependent 
X = data_prep.iloc[:,:-1]
Y = data_prep.iloc[:,-1]

# Split the dta into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = \
    train_test_split(X, Y,test_size=0.3,random_state=1234,stratify=Y)

# Import Random_Forest_Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=1234)
rfc.fit(x_train,y_train)
y_Predict = rfc.predict(x_test)

# Evaluate the confusion_matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_Predict)
score = rfc.score(x_test,y_test)

