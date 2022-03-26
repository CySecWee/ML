'''
50 start-up companies. This dataset contains five main information:
R&D Spend, Administration Spend, Marketing Spend, State, and Profit for
a financial year. Our goal is to create a model that can easily
determine which company has a maximum profit, and which is the most
affecting factor for the profit of a company.
'''

# importing libraries  
import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd

#importing datasets  
data_set= pd.read_csv('profit_data.csv')
print(data_set.RD[4],data_set.State[4] )

#Extracting Independent and dependent Variable
#https://www.shanelynn.ie/pandas-iloc-loc-select-rows-and-columns-dataframe/

x= data_set.iloc[:, 1:-1].values  
y= data_set.iloc[:, 5].values

print(x[4])
print(y[4])

''' last column contains categorical variables which are not suitable to
apply directly for fitting the model. So we need to encode this
variable.
'''
#https://towardsdatascience.com/categorical-encoding-using-label-encoding-and-one-hot-encoder-911ef77fb5bd
#https://www.educative.io/blog/one-hot-encoding

#Catgorical data - #Encode State Column 
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_x= LabelEncoder()  
x[:, 3]= labelencoder_x.fit_transform(x[:,3])
ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')
x = ct.fit_transform(x)
'''

print(x[0])
print(x[1])
print(x[2])
print(x[3])
print(x[4])
print()


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

A = make_column_transformer(
    (OneHotEncoder(categories='auto'), [3]), 
    remainder="passthrough")

x=A.fit_transform(x)

print(x[0])
print(x[1])
print(x[2])
print(x[3])
print(x[4])
print()

#avoiding the dummy variable trap:  
x = x[:, 1:]
print(x[0])
print(x[1])
print(x[2])
print(x[3])
print(x[4])
print()

# Splitting the dataset into training and test set.
''' if we don't specify the random_state, then every time
we run the code the train and test datasets would have different values each time.
'''
from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test= train_test_split(x, y, test_size= 0.3, random_state=0)


#Fitting the MLR model to the training set:  
from sklearn.linear_model import LinearRegression  
regressor= LinearRegression()  
regressor.fit(x_train, y_train)

#Predicting the Test set result;  
y_pred= regressor.predict(x_test)


print('Train Score: ', regressor.score(x_train, y_train))  
print('Test Score: ', regressor.score(x_test, y_test))

'''
Train Score:  91.7% accurate
Test Score:  82% accurate

'''


