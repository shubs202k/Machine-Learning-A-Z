#Multiple Linear Regression

#Data Preprocessing

#Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the Dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values #take all columns except last one
Y = dataset.iloc[:,4].values #take values of last column

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
(X[:,3]) = labelencoder_X.fit_transform(X[:,3])
#We have to avoid the algorithms to think one value in a column is greater 
#than another
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap
X = X[:,1:]

#Splitting the Dataset into Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,
                                                 random_state=0)

#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""

#Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#Predicting the Test Set Results
Y_pred = regressor.predict(X_test)

#Building the optimal model using backward elimination
import statsmodels.formula.api as sm #This library doesn't consider constant b0
X = np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
#Added a column of Ones which is the first column of X

#X_opt will have team of independent variables that will only have variables
##that have a considerable impact on the dependent variable
X_opt =X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=Y,exog=X_opt).fit()  #Initialize X_opt

#Now looking for the independent variable with highest p-value
regressor_OLS.summary()

#Removing X2 as it has highest p value
X_opt =X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

#Removing X1 as it has highest p value
X_opt =X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

#Removing Admin Spend as it has highest p value
X_opt =X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

#Removing Marketing Spend as it has highest p value
X_opt =X[:,[0,3]]
regressor_OLS = sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()