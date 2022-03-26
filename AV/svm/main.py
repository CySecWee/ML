#https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/

#Import Library
from sklearn import svm
import pandas as pd
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

#Load Train and Test datasets
#Identify feature and response variable(s) and values must be numeric and numpy arrays
train=pd.read_csv('train.csv')
train_y=train['Loan_Status']
train_x=train.drop(["Loan_Status"],axis=1)

test=pd.read_csv('test.csv')
test_y=test['Loan_Status']
test_x=test.drop(["Loan_Status"],axis=1)

# Create Linear SVM object
support = svm.LinearSVC(random_state=20)

# Train the model using the training sets and check score on test dataset
support.fit(train_x, train_y)
predicted= support.predict(test_x)
score=accuracy_score(test_y,predicted)
print("Your Model Accuracy is", score)
train.to_csv( "pred.csv")
