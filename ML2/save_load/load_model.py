# Load iris dataset from sklearn library
from sklearn.datasets import load_iris
import pickle
 

#laod the model from pickle file.

filename = "rf_model1.pickle"
with open(filename, "rb") as f: 
     model = pickle.load(f)

#get some predictions using loaded model.
print (model)

SepalLength = 6.3
SepalWidth = 2.3
PetalLength = 4.6
PetalWidth = 1.3

p=model.predict([[SepalLength, SepalWidth, PetalLength, PetalWidth]])

print(p)

