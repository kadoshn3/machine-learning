'''Machine Learning
By: Neeve Kadosh
Objective: Fit a linear regression model using a gradient descent
approach to noisy data
'''
import numpy as np
from plot_model import plot_model
from DesignFlags import DesignFlags
from gradient_descent import gradient_descent

# 1.0) Import design flags
design_flags = DesignFlags()

# 1.1) Noise to add to linear regression model
noise = design_flags.noise_multiplier * np.random.normal(0, 1, design_flags.m)

# 1.2) Declare variables
x_o = []
model = []
model_noise = []

# 1.3) Setup matrices
for idx in range(design_flags.m):
    x_o.append(idx)
    model.append(design_flags.slope*x_o[idx] + design_flags.y_int)
    model_noise.append(model[idx] + noise[idx])
    
# 2.0) Gradient descent computation
weight, cost = gradient_descent(x_o, model_noise)

# 2.1) Initialize new linear model
new_lin_model = []
for idx in range(len(x_o)):
    new_lin_model.append(weight[0] * x_o[idx] + weight[1])
equation = 'y = ' + str(weight[0]) + 'x + ' + str(weight[1])

# 3.0) Send model to be plotted with and without noise
plot_model(x_o, model, model_noise, new_lin_model, cost, equation)
