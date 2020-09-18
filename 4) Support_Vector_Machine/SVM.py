# Support_Vector_Machine
#--------------------------------------------------

# Import Libraries
import pandas as pd

# Read the data and create a Copy
LoanData = pd.read_csv("L:/Data_Science/Machine_Learning/Support_Vector_Machine/01Exercise1.csv")

LoanPrep = LoanData.copy()

# Identify the Missing Values 
LoanPrep.isnull().sum(axis=0)

# Drop the row with Missing values
LoanPrep = LoanPrep.dropna()

LoanPrep = LoanPrep.drop(["gender"],axis=1)

# Create Dummy variable for Catagorical variable
LoanPrep.dtypes
LoanPrep = pd.get_dummies(LoanPrep,drop_first=True)

# Normalize the continuous Data (Income and Loanamt) using StandardScaler
from sklearn.preprocessing import StandardScaler
Scaler_ = StandardScaler()
LoanPrep["income"] = Scaler_.fit_transform(LoanPrep[["income"]])
LoanPrep["loanamt"] = Scaler_.fit_transform(LoanPrep[["loanamt"]])

# Create X and Y
Y = LoanPrep[["status_Y"]]
X = LoanPrep.drop(["status_Y"],axis=1)

# Split the dataset into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =  \
    train_test_split(X, Y, test_size=0.3,random_state=1234,stratify=Y)
    
# Create the SVM Classification
from sklearn.svm import SVC
svc = SVC()

svc.fit(x_train,y_train)
 
y_predict = svc.predict(x_test)

# Bulid The Confusion matrix and get the accuary/score
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predict)
svc.score(x_test,y_test)

# Accuracy , Precision , AUC-ROC







