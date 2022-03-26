#https://www.analyticsvidhya.com/blog/2021/09/a-complete-guide-to-understand-classification-in-machine-learning/
from numpy import where
from collections import Counter
from sklearn.datasets import make_blobs
from matplotlib import pyplot

X, y = make_blobs(n_samples=1000, centers=5, random_state=1)
print(X.shape, y.shape)
counter = Counter(y)
print(counter)
for i in range(10):
  print(X[i], y[i])
for label, _ in counter.items():
  row_ix = where(y == label)[0]
  pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()
