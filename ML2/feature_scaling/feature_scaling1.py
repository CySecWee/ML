#https://deepblade.com/artificial-intelligence/machine-learning/why-use-feature-scaling/

#Normalization

import numpy as np
data = np.array([[26, 50000],
             [29, 70000],
             [34, 55000],
             [31, 41000]])

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
print(data)
print(scaled_data)

