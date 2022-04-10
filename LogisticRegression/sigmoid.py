import numpy as np

def sigmoid(w, x):
    return 1/(1+np.exp(-w.T * x))