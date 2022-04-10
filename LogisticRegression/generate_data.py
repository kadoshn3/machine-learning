from DesignFlags import DesignFlags
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
# Generates 2 feature, 2 class data
# returns x and y for 2 classes
def generate_data():
    # Import design flgas
    design_flags = DesignFlags()
    
    # make 2 class data
    x, y = make_blobs(n_samples=design_flags.m, n_features=2, centers=2)
    
    # Plot 2 class data set
    plt.figure(1)
    plt.scatter(x[:, 0], x[:, 1], c = y)
    plt.xlabel('X Feature')
    plt.ylabel('Y Feature')
    plt.title('Blobs in Need of Classifying')
    plt.grid()
    
    plt.show()
    
    return x, y