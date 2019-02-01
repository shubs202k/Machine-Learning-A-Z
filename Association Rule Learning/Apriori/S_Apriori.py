# Apriori

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib qt

# Importing the dataset
# Header = None to specify that there are no titles in dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv',header = None)

# Apriori Model expects dataset to be a list of lists
# Preparing the Input
# Apriori model expects input to be a string

transactions =[]
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
    
# Training Apriori on the Dataset
from apyori import apriori
rules = apriori(transactions,min_support = 0.003 ,
                min_confidence = 0.2 ,min_lift = 3, min_length=2) 

# min_support = Rules to consider products that are bought atleast 
# 3 times a day   
# min_length =2 indicates that rules will have atleast 2 products
# If min_confidence =  default value of 0.8 means rules have to be correct
# 80% of the time

# Vizualizing the Results
results = list(rules)
results_list = []
for i in range(0, len(results)):
    results_list.append('RULE:\t' + str(results[i][0]) +
                        '\nSUPPORT:\t' + str(results[i][1]))















