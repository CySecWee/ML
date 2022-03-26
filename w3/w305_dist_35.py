import numpy
import matplotlib.pyplot as plt

x = numpy.random.normal(1.0, 1.0, 1000)

plt.hist(x, 100)
plt.show()
