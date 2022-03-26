
'''
Backward elimination is a feature selection technique while building
a machine learning model. It is used to remove those features that do
not have a significant effect on the dependent variable or prediction of
output.
'''
#https://www.javatpoint.com/backward-elimination-in-machine-learning

# importing libraries  


import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd  
  
#importing datasets  
data_set= pd.read_csv('profit_data.csv')  

#Extracting Independent and dependent Variable  

x= data_set.iloc[:, 1:-1].values  
y= data_set.iloc[:, 5].values


#Catgorical data  

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

A = make_column_transformer(
    (OneHotEncoder(categories='auto'), [3]), 
    remainder="passthrough")

x=A.fit_transform(x)

#avoid dummy variable trap
x = x[:, 1:]


# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.3, random_state=0)  
  
#Fitting the MLR model to the training set:  
from sklearn.linear_model import LinearRegression  
regressor= LinearRegression()  
regressor.fit(x_train, y_train)


#Predicting the Test set result;  
y_pred= regressor.predict(x_test)  
  
#Checking the score
rs_train=regressor.score(x_train, y_train)
rs_test=regressor.score(x_test, y_test)

print('Train Score: ', rs_train)  
print('Test Score: ', rs_test)

df=(rs_train-rs_test) * 100

print ('the difference : ' , df)

import statsmodels.api as smf

x = nm.append(arr = nm.ones((50,1)).astype(int), values=x, axis=1)

#apply a backward elimination proces
x_opt=x [:, [0,1,2,3,4,5]]  
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()  
regressor_OLS.summary()



