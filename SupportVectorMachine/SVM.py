import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal as mvn
from sklearn.model_selection import train_test_split
from qpsolvers import solve_qp
from sklearn import svm

'''Support Vector Machine
By: Neeve Kadosh
Date: 17 October 2019
'''

plt.close('all')

# Sample size
sample_size = 1000

# Covariance
covariance = [[5, 50],[50, 5]]

# Mean first class
mean_0 = [40, 40]
mean_1 = [5, 5]

# Generate 2 class gaussian data set
x_0 = mvn(mean_0, covariance, sample_size)
x_1 = mvn(mean_1, covariance, sample_size)

# Creating labels, class 1 - 0s, class 2 - 1s
y_0 = np.zeros((np.shape(x_0)))
y_1 = np.ones((np.shape(x_1)))

# Concatenated data and labels
x = np.concatenate((x_0, x_1))
y = np.concatenate((y_0, y_1))

# Visualization of the gaussian data
debug = True
if debug:
    plt.figure
    plt.scatter(x_0[:,0], x_0[:,1], c='r', label='Unlabeled Class 1')
    plt.scatter(x_1[:,0], x_1[:,1], c='g', label='Unlabeled Class 2')
    plt.grid()
    plt.legend()
    plt.show()

# Split data into testing and training data
test_size = .2
x_train, x_test, y_train, y_test = \
            train_test_split(x, y, test_size=test_size)

P = np.dot(x_train, y_train.T)  + 1e-8 * np.eye(len(x_train))
print(np.shape(P))
q = -np.ones(len(x_train))
G = np.eye(len(x_train))
h = -1 * np.ones(len(x_train))

w = solve_qp(P, q, G, h)
print('Weights are:', w)
length = np.linspace(0, 1, len(w))
plt.scatter(length, w, marker='x')
