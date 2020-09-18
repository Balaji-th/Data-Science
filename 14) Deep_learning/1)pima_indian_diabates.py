# Neural Network using keras

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense 

# Read dataset
data = pd.read_csv('E:/Data_Science/Machine_Learning/14) Deep_learning/pima-indians-diabetes.csv')

data.isnull().sum(axis=0)

x = data.iloc[:,0:-1]
y = data.iloc[:,-1]

# Split X and Y
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.2, random_state=1234,stratify=y)
    
# Define the keras model
model = Sequential()

model.add(Dense(24,
                input_shape=(8,),
                activation='relu',
                kernel_initializer='RandomNormal'))

model.add(Dense(12,
                activation='relu',
                kernel_initializer='RandomNormal'))

model.add(Dense(1,
                activation='sigmoid'))

# Complie the model
model.compile(optimizer='adam',
               loss='binary_crossentropy',
               metrics=['accuracy'])

# run the model
model.fit(x_train,y_train,epochs=160,batch_size=10)

# Get the accuracy score to evaluate the model
accuracy_test = model.evaluate(x_test, y_test)

# Get the predicted values and predicted probabilies of Y_test
y_predict = model.predict_classes(x_test)
y_pred_prob = model.predict(x_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)











