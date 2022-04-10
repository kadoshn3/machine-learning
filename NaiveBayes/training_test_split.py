from sklearn.model_selection import train_test_split
from design_flags import DesignFlags
import matplotlib.pyplot as plt

def training_test_split(x, y):
    # Import design flags
    design_flags = DesignFlags()
    
    # Split data into training and testing data
    x_train, x_test, y_train, y_test = train_test_split( \
                                       x, y, test_size=1-design_flags.split)
    
    return x_train, x_test, y_train, y_test 