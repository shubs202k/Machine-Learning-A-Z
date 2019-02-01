#Data Preprocessing

#Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the Dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values #take all columns except last one
Y = dataset.iloc[:,3].values #take values of last column

#Splitting the Dataset into Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,
                                                 random_state=0)
#test_size=0.5 means half of our data goes into training set and other 
#half to test set
#test_size = 0.2 ,2 observations in test set and 8 in training set

#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""






    