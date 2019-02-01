#Data Preprocessing

#Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the Dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values #take all columns except last one
Y = dataset.iloc[:,3].values #take values of last column

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy = 'mean',axis=0)
#Fit imputer only to columns where there is some missing data
imputer = imputer.fit(X[:, 1:3]) #imputer object fitted to matrix X
#Replace missing data of matrix X by mean of columns
X[:, 1:3] = imputer.transform(X[:, 1:3])

#We need to encode text to numbers since machine learning uses math equations
#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
#Apply LabelEncoder object on column country,return encoded data
(X[:,0]) = labelencoder_X.fit_transform(X[:,0])

#We have to avoid the algorithms to think one value in a column is greater 
#than another
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

#Only use labelencoder as Purchased colum is dependent
labelencoder_Y = LabelEncoder()
#Apply LabelEncoder object on column purchased ,return encoded data
(Y) = labelencoder_Y.fit_transform(Y)

#Splitting the Dataset into Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,
                                                 random_state=0)
#test_size=0.5 means half of our data goes into training set and other 
#half to test set
#test_size = 0.2 ,2 observations in test set and 8 in training set

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)






    