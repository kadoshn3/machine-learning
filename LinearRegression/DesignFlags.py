class DesignFlags():
    def __init__(self):
        # m - sample size
        self.m = 500
        
        # Model parameters
        self.slope = 3
        self.y_int = 5
        
        # Noise modifier
        self.noise_multiplier = 5
        
        # Learning adjustments
        self.learning_rate = .1
        # slope initial guess
        self.w1 = 5
        # y_intercept initial guess
        self.w2 = 10
        
        self.stopping_value = 0