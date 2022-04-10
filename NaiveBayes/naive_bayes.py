import numpy as np
from design_flags import DesignFlags
from generate_data import generate_data
from training_test_split import training_test_split
import matplotlib.pyplot as plt
''' Naive Bayes Algorithm
parameters: 
    x: feature, list, size = m
    y: feature, list, size = m
    
    x_train: feature, list, size = m * split
    y_train: feature, list, size = m * split
    
    x_test: feature, list, size = m * (1-split)
    y_test: feature, list, size = m * (1-split)
    
    cov: covariance, shape = 2
    mean: mean, shape = 2
    
    likelihood: likelihood, matrix:
                rows = len(x_train), columns = 2
                
    prob_class: scalar, probability of class occuring
    prob_xy: scalar, probability of feature occuing
    
    posterior: posterior probability, matrix:
                rows = len(x_train), columns = 2
                
    minimum: min of x and y posterior, shape = 2
    maximum: max of x and y posterior, shape = 2
    separation: average of min and max, shape = 2
    
    classifier: labels data to class with 1 or 0, matrix:
                rows = len(x_train), columns = 2
    
    xy_label_12: class labels for xy training data, list, 
                 size depends on classifier result
hyperparamters:
    m: scalar, number of data points
'''
# Closes all open plots
plt.close('all')

# Import design flags
design_flags = DesignFlags()

# Generated dual feature data for 2 class problem
x, y = generate_data()

# Split data into training and testing data
x_train, x_test, y_train, y_test = training_test_split(x, y)

# Shape of data
shape = np.shape(x_train)

# Covariance dxd
cov = np.cov(x_train.T)
print(cov)
# Mean dxd
mean = np.mean(x_train.T)
print(mean)
# Likelihood 
likelihood = np.zeros((shape))
for j in range(shape[1]):
    for i in range(shape[0]):
        likelihood[i, j] = (1 / (np.sqrt(2 * np.pi) * cov)) *       \
                (np.exp(-(x_train[i, j] - mean) / (cov * np.sqrt(2)))) ** 2

# Probability of class and data occurances
prob_class = .5

# Posterior probability
posterior = likelihood * prob_class
plt.figure()
plt.grid()
plt.show()
'''
# Classfies labels as 1 or 0
classifier = np.zeros((len(posterior), 2))
for idx in range(len(posterior)):
    if posterior[idx, 0] > separation[0]:
        classifier[idx, 0] = 1
    if posterior[idx, 1] > separation[1]:
        classifier[idx, 1] = 1

# Initialize and assign labels
x_label_1 = []
x_label_2 = []
y_label_1 = []
y_label_2 = []

for idx in range(len(classifier)):
    if classifier[idx, 0] == 1:
        x_label_1.append(x_train[idx])
    else:
        x_label_2.append(x_train[idx])
        
    if classifier[idx, 1] == 1:
        y_label_1.append(y_train[idx])
    else:
        y_label_2.append(y_train[idx])


# Plot labels
plt.figure()
plt.scatter(x_label_1, y_label_1, label='Class 1')
plt.scatter(x_label_2, y_label_2, label='Class 2')
plt.xlabel('X Labeled Data')
plt.ylabel('Y Labeled Data')
plt.title('Labled Gaussian Blobs')
plt.legend()
plt.grid()
plt.show()
'''

