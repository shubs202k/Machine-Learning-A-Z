#Natural Language Processing

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib qt
# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t',quoting = 3)
# quoting = 3 avoids any problems caused by double quotes

# Cleaning the Data
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i]) # Dont remove all letters,remove everything else in review column and replace by space
    review = review.lower() # All characters in lower case
    review = review.split() # will split the string of words into a list
    ps = PorterStemmer() # Object of class PorterStemmer created,# Remove non significant words like the,and etc
    review = [ps.stem(word) for word in review if word not in 
              set(stopwords.words('english'))] # word 'this' removed ,# stemming === keep only the root of the word. For eg loved = love,loving etc
    review = ' '.join(review) # Convert review back to a string from a list
    corpus.append(review)
    
# Creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) # Excluding non relevant words   
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
 
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes Model to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)









