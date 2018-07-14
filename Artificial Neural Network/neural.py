# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data = pd.read_csv('Churn_Modelling.csv')
X = data.iloc[:, 3:13].values
y = data.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_x1 = LabelEncoder()
X[:, 1] = le_x1.fit_transform(X[:, 1])
le_x2 = LabelEncoder()
X[:, 2] = le_x2.fit_transform(X[:, 2])
ohe = OneHotEncoder(categorical_features = [1])
X = ohe.fit_transform(X).toarray()
#to escape dummy variable trap
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense
ANN = Sequential()
ANN.add(Dense(output_dim = 6, activation = 'relu', init = 'uniform', input_dim = 11))
ANN.add(Dense(output_dim = 6, activation = 'relu', init = 'uniform'))
ANN.add(Dense(output_dim = 1, activation = 'sigmoid', init = 'uniform'))
ANN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

ANN.fit(X_train, y_train, batch_size= 10, epochs = 100)