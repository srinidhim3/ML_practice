#Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
#print(tf.__version__)

#Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values

#Encoding the categorical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# 1 is the index of the column.
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#splitting the data into test and training set.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Feature Scaling is mandatory for deep learning.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Building ANN
#Initialize
ann = tf.keras.models.Sequential()
#Adding input layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
#Adding second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
#Adding ouput layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#Training the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(X_train, y_train, batch_size=32, epochs=100)

#Making the prediction and evaluating the model
pred = ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5
print(pred)

y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(y_pred)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
ac = accuracy_score(y_test, y_pred)
print(ac)
'''
[[1504   91]
 [ 183  222]]
0.863
'''