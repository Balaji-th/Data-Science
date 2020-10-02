import pandas as pd
dataset=pd.read_csv('loan_small.csv')
cleandata = dataset.dropna(subset=["Loan_Status"])

df = cleandata.copy()
cols=["Gender","Area","Loan_Status"]
df[cols] = df[cols].fillna(df.mode().iloc[0])

cols2=["ApplicantIncome","CoapplicantIncome","LoanAmount"]
df[cols2] = df[cols2].fillna(df.mean())

df = df.drop(["Loan_ID"],axis=1)
#dfbedum = df.copy()

df = pd.get_dummies(df,drop_first=True)

#Split the data Vertically
x = df.iloc[:,:-1]  #expect loan status
y = df.iloc[:,-1]  #only last column loan status

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = \
  train_test_split(x,y,test_size=0.2,random_state=1234)

