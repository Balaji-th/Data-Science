# IRIS Plant Classification
#------------------------------------------------------

# Import Libraries
 from sklearn import datasets

iris =datasets.load_iris() 

x = iris.data
y = iris.target

# Split into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = \
    train_test_split(x, y,test_size=0.3,random_state=1234,stratify=y)

# Train the SVC 
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# rbf kernal with gamma as 1
svc = SVC(kernel='rbf',gamma=1.0)
svc.fit(x_train,y_train)
y_predict = svc.predict(x_test)
cm_rbf01 = confusion_matrix(y_test, y_predict)

# rbf kernal with gamma as 10
svc = SVC(kernel='rbf',gamma=10.0)
svc.fit(x_train,y_train)
y_predict = svc.predict(x_test)
cm_rbf10 = confusion_matrix(y_test, y_predict)

# Linear kernal
svc = SVC(kernel='linear')
svc.fit(x_train,y_train)
y_predict = svc.predict(x_test)
cm_linear = confusion_matrix(y_test, y_predict)

# poly kernal
svc = SVC(kernel='poly')
svc.fit(x_train,y_train)
y_predict = svc.predict(x_test)
cm_poly = confusion_matrix(y_test, y_predict)

# Sigmoid
svc = SVC(kernel='sigmoid')
svc.fit(x_train,y_train)
y_predict = svc.predict(x_test)
cm_sig = confusion_matrix(y_test, y_predict)















