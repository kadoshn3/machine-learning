from DesignFlags import DesignFlags
import numpy as np
from cost_function import cost_function

def gradient_descent(x, model_noise):
    # Import design flags
    design_flags = DesignFlags()
    
    # Initialize weights
    old_weight = [design_flags.w1, design_flags.w2]
    new_weight = []
    # Initialize cost
    cost = []
    # To begin the data so it runs first time
    threshold = [10000000, 10000000]
    idx = 0
    # Loop to find gradient descent
    # If the old weight is less than the stopping value,
    # Then the algorithm will exit the loop
    while (threshold[0] > design_flags.stopping_value) & (threshold[1] > design_flags.stopping_value): 
        # new_weights = old_weights - learning rate * 1/m * sum(dJ(x)/dx)
        new_weight = old_weight - (design_flags.learning_rate / design_flags.m) \
                    * (np.dot(old_weight, x[idx]) - model_noise[idx]) * x[idx]
                    
        if idx != 0:
            threshold = (np.dot(new_weight, x[idx]) - model_noise[idx]) * x[idx]
        
        print('m_'+ str(idx), '=', str(old_weight[0]),  ' b_'+ str(idx), '=', str(old_weight[1]))
        
        cost.append(cost_function(old_weight, x, model_noise, idx))
        
        old_weight = new_weight
        
        idx += 1
        
    return new_weight, cost