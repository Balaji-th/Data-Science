# import libraries
import pandas as pd

# Read the data and Create a copy
loandata = pd.read_csv("L:/Data_Science/Machine_Learning/7) Evaluate_Classification_Models/01Exercise1.csv")

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

# Predict using Test Data
y_predict = lr.predict(x_test)

# Build the confusion matrix and get the accuracy/score
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_predict)

lr.score(x_test, y_test)

cr = classification_report(y_test, y_predict)

# Accuracy score
from sklearn.metrics import accuracy_score

score2 = accuracy_score(y_test, y_predict)

# Adjust the Probabilities
y_prob = lr.predict_proba(x_test)[:,1]

# Classsification based on probability values
y_new_pred = []
threshold = 0.7555
for i in range(0,len(y_prob)):
    if y_prob[i] > threshold:
        y_new_pred.append(1)
    else:
        y_new_pred.append(0)

# Build the confusion matrix and get the accuracy/score
cm2 = confusion_matrix(y_test, y_new_pred)
score2 = lr.score(x_test, y_new_pred)
cr2 = classification_report(y_test, y_new_pred)

# Understend and implement AUC ROC Curve
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, threshold=roc_curve(y_test,y_prob)
auc = roc_auc_score(y_test, y_prob)

# Plot the ROC
import matplotlib.pyplot as plt

plt.plot(fpr,tpr, linewidth=4)
plt.xlabel("Flase Positive rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Loan Prediction")
plt.grid()
