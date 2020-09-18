# import libraries
import pandas as pd

# Read the data and Create a copy
loandata = pd.read_csv("L:/Data_Science/Machine_Learning/Logistic_Regression/01Exercise1.csv")

loanprep = loandata.copy()

# Identify the missing values
loanprep.isnull().sum(axis=0)

# Drop the rows with missing data
loanprep = loanprep.dropna()

loanprep = loanprep.drop(["gender"],axis=1)

# Create the dummy variable for Categorical variable
loanprep.dtypes
loanprep = pd.get_dummies(loanprep,drop_first=True)

# Normalize the data (Income and Loanamount) using StandardScalar
from sklearn.preprocessing import StandardScaler
scalar_ = StandardScaler()

loanprep["income"] = scalar_.fit_transform(loanprep[["income"]])

loanprep["loanamt"] = scalar_.fit_transform(loanprep[["loanamt"]])

# Create X and Y variable
Y =loanprep[["status_Y"]]
X = loanprep.drop(["status_Y"],axis=1)

# Split the dataset to train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = \
    train_test_split(X, Y, test_size=0.3, random_state=1234, stratify=Y)
    
# Create the Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)

y_predict = lr.predict(x_test)

# Build the confusion matrix and get the accuracy/score
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)

lr.score(x_test, y_test)




