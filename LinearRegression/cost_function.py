from DesignFlags import DesignFlags
import numpy as np

def cost_function(weight, x_o, model_noise, idx):
    design_flags = DesignFlags()
    m = design_flags.m
    
    # Mean square error
    cost = (1 / (2*m)) * ((np.dot(weight, x_o[idx]) - model_noise[idx]) ** 2)
    
    return cost