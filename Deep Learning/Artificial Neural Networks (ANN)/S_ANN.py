# ANN

# Part 1 : Data Preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the Dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values 
Y = dataset.iloc[:, 13].values 

# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X_1 = LabelEncoder()
(X[:,1]) = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
(X[:,2]) = labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[: , 1:]

# Spliting Dataset into Training and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,
                                                 random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

# Part 2 : Now lets make the ANN
# Importing keras libraries and packages

import keras
from keras.models import Sequential
from keras.layers import Dense # Add layers to ANN

# Initializing the ANN
classifier = Sequential() 

# Adding the Input Layer and Hidden Layer
classifier.add(Dense(6, input_dim = 11 ,
                     kernel_initializer = 'uniform'
                     , activation = 'relu'))

# Number of nodes in hidden layer 11 + 1 = 12/2 = 6
# Input dimension must be specifed to indicate from where 
# the hidden layer is expecting input from
# activation function used is retifier which is used for Input layers
# we use sigmoid function for output layers

# Adding the second hidden layer
# No need to mention input_dim as second hiiden layer knows
# from where to expect the input i.e from first hidden layer
classifier.add(Dense(6 , kernel_initializer = 'uniform'
                     , activation = 'relu'))

# Adding the Output layer, we want only one node in output layer
classifier.add(Dense(1 , kernel_initializer = 'uniform'
                     , activation = 'sigmoid'))

# If dependent variable is of more than 2 categories
# output_dim = 3 and activation = softmax

# Compiling the ANN === Adding stochastic gradient descent
classifier.compile(optimizer = 'adam', 
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])
# metrics ====== When weights are updated after each observation
#                the algo uses the accuracy criterion to improve
#                the model performance                   
# optimizer ==== algorithm to use to find the optimal set of
#                weights in the neural network

# Fitting ANN to the Training set
classifier.fit(X_train,Y_train,batch_size = 10 ,nb_epoch = 100)

# Making the predictions
Y_pred = classifier.predict(X_test) # Probability that person leaves
Y_pred =(Y_pred > 0.5) # To get results in form of true/false

#Making the Confusion Matrix to check our predictions
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred) # Accuracy 84.4%






