'''
We can’t represent 64 dimensions data set as a
single plot. Therefore, we can use principal component analysis (PCA) to
reduce the number of dimensions to a level that can be represented here.
Here, we will reduce the number of dimensions to 2.
'''

# load data set from sklearn library
from sklearn.datasets import load_digits
digits = load_digits()

#see the dimensions

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
new_data = pca.fit_transform(digits.data)

# The shape of the data set before using PCA
print(digits.data.shape)

# The shape of the data set after using PCA
print(new_data.shape)

#We can visualise those components as follows.

import matplotlib.pyplot as plt
plt.scatter(new_data[:, 0], new_data[:, 1], c=digits.target)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.colorbar()
