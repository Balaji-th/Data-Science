from sklearn.datasets import load_breast_cancer
import pandas as pd

lbc = load_breast_cancer()

x = pd.DataFrame(lbc["data"],columns=lbc["feature_names"])
y = pd.DataFrame(lbc["target"],columns=["type"])

#---------------------------------------------------------
# Classification Without PCA
#--------------------------------------------------------

# Split the detaset into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = \
    train_test_split(x, y,test_size=0.3,random_state=1234,stratify=y)

# Import Random Forest Classsification and train_model (RFC)
from sklearn.ensemble import RandomForestClassifier
rfc1 = RandomForestClassifier(random_state=1234)
rfc1.fit(x_train, y_train)
y_predict1 = rfc1.predict(x_test)

# Confusion_Matrix,Score and Evaluate the Model
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_predict1)

score1 = rfc1.score(x_test, y_test)

#--------------------------------------------------------
# Implement PCA
#----------------------------------------------------------

# Center the data
from sklearn.preprocessing import StandardScaler
scaler_ = StandardScaler()
x_scaled = scaler_.fit_transform(x)

# Create PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
x_pca = pca.fit_transform(x_scaled)

# Implement PCA in DATA
x_train, x_test, y_train, y_test = \
    train_test_split(x_pca, y,test_size=0.3,random_state=1234,stratify=y)

# Import Random Forest Classsification and train_model (RFC)
rfc2 = RandomForestClassifier(random_state=1234)
rfc2.fit(x_train, y_train)
y_predict2 = rfc2.predict(x_test)

# Confusion_Matrix,Score and Evaluate the Model
cm2 = confusion_matrix(y_test, y_predict2)
score2 = rfc2.score(x_test, y_test)