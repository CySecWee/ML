# Load iris dataset from sklearn library
from sklearn.datasets import load_iris
dataset = load_iris()

# Split dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size = 0.2)

# import RandomForestClassifier and create a model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)

# train the model
rf.fit(X_train, y_train)

# Calculate accuracy of the model
print("Accuracy : {}".format(rf.score(X_test, y_test)))

''' Pickle is a standard Python module that can used for serializing and
de-serializing python object structures.
'''

import pickle
 

#save the model as a pickle file.

filename = "rf_model1.pickle"
with open(filename, "wb") as f:
     pickle.dump(rf, f)

