# load data set from sklearn library
from sklearn.datasets import load_digits
digits = load_digits()

#see the dimensions

print(digits.data.shape)

#A data point represented here is an 8×8 pixels image. That is 64 dimensions.

# Visualise a single data point
import matplotlib.pyplot as plt
plt.matshow(digits.images[1])
plt.show()

