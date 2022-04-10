import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

# Plot of model with and without noise
def plot_model(x_o, model, model_noise, new_lin_model, cost, equation):
    plt.figure(1)
    plt.plot(x_o, model, label='y = 3x', color='r')
    plt.plot(x_o, new_lin_model, label=equation, color='g')
    plt.scatter(x_o, model_noise, label='Linear Model with Noise')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title('Linear Regression Model')
    plt.legend()
  
    '''
    plt.figure(2)
    x = np.arange(0, len(cost))
    plt.scatter(x, cost)
    plt.title('Cost')
    plt.ylabel('Mean Square Error')
    '''
    plt.show()
    