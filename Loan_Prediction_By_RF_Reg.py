# -*- coding: utf-8 -*-
# Data preprocessing 
# importing libraries

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

# The following classe is to manager missing values in categorical data
from sklearn.base import TransformerMixin, BaseEstimator

class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean',filler='NA'):
       self.strategy = strategy
       self.fill = filler

    def fit(self, X, y=None):
       if self.strategy in ['mean','median']:
           if not all(X.dtypes == np.number):
               raise ValueError('dtypes mismatch np.number dtype is \
                                 required for '+ self.strategy)
       if self.strategy == 'mean':
           self.fill = X.mean()
       elif self.strategy == 'median':
           self.fill = X.median()
       elif self.strategy == 'mode':
           self.fill = X.mode().iloc[0]
       elif self.strategy == 'fill':
           if type(self.fill) is list and type(X) is pd.DataFrame:
               self.fill = dict([(cname, v) for cname,v in zip(X.columns, self.fill)])
       return self
   
    def transform(self, X, y=None):
        return X.fillna(self.fill)
    
# Preprocessing the training set 
# Importing dataset
dataset =  pd.read_csv('train_u6lujuX_CVtuZ9i.csv')

# Taking of data set by replaccing with the most frequent
dataset = CustomImputer(strategy='mode').fit_transform(dataset)

#seperating dependent variable and independent variable 
X = dataset.iloc[:, 1:12].values 
Y = dataset.iloc[:, 12].values
 

#categorization
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
collection = [0,1,2,3,4,10]
for i in collection:
     X[:, i] = label_encoder.fit_transform(X[:, i])
     Onehotencoder = OneHotEncoder(categorical_features = [i])
X = Onehotencoder.fit_transform(X).toarray()

# Encoding the Dependent Variable
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#Preprocessing the test set
#==============================================================================
# 
#==============================================================================
#Preprocessing the training set 
# Importing dataset
test_dataset =  pd.read_csv('test_Y3wMUE5_7gLdaTN.csv')

# Takaking care of missing data in the test set
test_dataset = CustomImputer(strategy='mode').fit_transform(test_dataset)

# Selecting important dependent variable  
X_test = test_dataset.iloc[:, 1:12].values 
 
#categorization
# Encoding categorical data for the test set
# Encoding the Independent Variable for the test data set
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
collection = [0,1,2,3,4,9,10]
for i in collection:
     X_test[:, i] = label_encoder.fit_transform(X_test[:, i])
     Onehotencoder = OneHotEncoder(categorical_features = [i])
X_test = Onehotencoder.fit_transform(X_test).toarray()
               
# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, Y)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
