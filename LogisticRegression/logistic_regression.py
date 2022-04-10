from generate_data import generate_data
from training_test_split import training_test_split
from DesignFlags import DesignFlags
import numpy as np
import matplotlib.pyplot as plt

# Sigmoid activation
def sigmoid(w, x):
    h = 1/(1+np.exp(-w * x))
    return h

# Import design flags
design_flags = DesignFlags()

# Generated dual feature data for 2 class problem
x, y = generate_data()

# Split data into training and testing data
x_train, x_test, y_train, y_test = training_test_split(x, y)

# Initialize weight
w_old = design_flags.w

# new_w = old_w - lr/m dJ(x)
# dJ(x) = (sigmoid(w, x.T) - x.T)x.T
for idx in range(len(x_train)):                # (sigmoid(x @ w) - y) * x[idx]
    J_x = (sigmoid(w_old, x_train[idx, 0]) - x_train[idx, 1]) * x_train[idx, 0]
    w_new = w_old - (design_flags.learning_rate/design_flags.m) * J_x
    
    print('Weight', str(idx), '=', w_new)
    w_new = w_old

step = max(x_train[:,0]) / design_flags.m
x_axis = np.arange(min(x_train[:,0]), max(x_train[:,0]), step)
model = 1 / (1+np.exp(-w_new*x_axis))
plt.plot(x_axis, model)

