class DesignFlags():
    def __init__(self):
        # m - sample size
        self.m = 1000
        
        # Train = split | Test = 1 - split
        self.split = .8

        self.learning_rate = .7
        
        self.w = 1