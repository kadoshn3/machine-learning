from design_flags import DesignFlags
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
    plt.subplot(211)
    plt.scatter(x[:, 0], x[:, 1], c = y)
    plt.xlabel('X Feature')
    plt.ylabel('Y Feature')
    plt.title('Blobs in Need of Classifying')
    plt.grid()
    
    hspace = .4
    plt.subplots_adjust(hspace=hspace)
    plt.subplot(212)
    n_bins = int(design_flags.m * .6)
    plt.hist(x[:, 0], bins=n_bins, label='2 Class X-Feature')
    plt.hist(x[:, 1], bins=n_bins, label='2 Class Y-Feature')
    plt.title('Feature Histogram')
    plt.grid()
    plt.legend()
    
    plt.show()
    
    return x, y