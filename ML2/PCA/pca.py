''' Principal component analysis (PCA) is a dimensionality reduction
technique that can used in Unsupervised Learning. It is important to
minimise the number of dimensions when working with a data set that has
a large number of features.

'''
#https://deepblade.com/artificial-intelligence/machine-learning/principal-component-analysis-pca/

import numpy as np
import matplotlib.pyplot as plt

# create dataset as numpy array
data = np.array([[40, 20],
                [55, 30],
                [70, 60],
                [50, 35],
                [45, 40],
                [62, 75],
                [45, 30],
                [68, 80],
                [80, 70],
                [75, 90]])
n=len(data)
# feature scaling

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

#for i in range(n):
#    print(data[i][0], scaled_data[i][0], data[i][1], scaled_data[i][1] )

print(scaled_data)

''' create a covariance matrix to measure the relationships between the
features. Then we can identify the principal components (PCs) by
calculating eigenvalues and eigenvectors using the covariance matrix.

the number of principal components equal to the number of dimensions of
the problem. this example gives a maximum of two
principal components.
'''

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(data)


print(pca.explained_variance_)
print(pca.explained_variance_ratio_)

#number of dimensions in the data set has been reduced from 2 to 1.

data_pca = pca.transform(data)
print("Original shape:", data.shape)
print("Transformed shape:", data_pca.shape)

new_data = pca.inverse_transform(data_pca)
plt.scatter(data[:, 0], data[:, 1], alpha=0.2)
plt.scatter(new_data[:, 0], new_data[:, 1])
plt.show()


