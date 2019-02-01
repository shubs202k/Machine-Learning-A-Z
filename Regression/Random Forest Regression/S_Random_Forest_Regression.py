#Random Forest Regression

#Random Forest is a team of decision trees wherein each tree 
#makes a prediction and the final prediction is a average of 
#all tree predictions

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Random Forest Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300,random_state = 0)
#n_estimators is the number of decision trees
regressor.fit(X,Y)

#Predicting a new result with Regression for n_estimators = 300
Y_pred = regressor.predict([[6.5]])

# Visualising the Random Forest Model results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
#Number of Stairs increase compared to Decision Tree Model
#The more we add trees better the prediction





