import numpy as np
from prettytable import PrettyTable

class Maze_Class():
    def __init__(self):
        
        # Select maze difficulty
        # Options: Easy, Medium, Hard
        self.difficulty = 'Hard'
        
        # Maze environment setup
        #
        # Hard maze environment 20x20
        if self.difficulty == 'Hard':
            self.env = np.array([
                [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  1., 1.,  1.,  0.,  1.,  1.,  1.,  0.,  1.,  1.,  1.],
                [ 1.,  1.,  1.,  0.,  0.,  1.,  0.,  1.,  0.,  1., 0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.],
                [ 0.,  0.,  0.,  0.,  1.,  1.,  0.,  1.,  1.,  1., 0.,  1.,  0.,  1.,  0.,  1.,  1.,  1.,  0.,  1.],
                [ 1.,  0.,  1.,  1.,  1.,  0.,  0.,  1.,  0.,  0., 0.,  1.,  1.,  1.,  0.,  1.,  0.,  0.,  0.,  1.],
                [ 1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1., 1.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  1.],
                [ 0.,  0.,  1.,  0.,  1.,  1.,  1.,  0.,  1.,  1., 1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  1.],
                [ 0.,  1.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  0., 0.,  1.,  1.,  1.,  0.,  0.,  1.,  1.,  1.,  1.],
                [ 0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  1.,  1., 0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  0.,  0.],
                [ 0.,  1.,  1.,  1.,  1.,  0.,  1.,  0.,  1.,  1., 1.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  1.,  1.],
                [ 0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  0., 0.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.,  0.],
                [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,  1., 1.,  1.,  0.,  1.,  0.,  0.,  1.,  1.,  0.,  0.],
                [ 0.,  1.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,  0., 0.,  1.,  0.,  1.,  1.,  0.,  1.,  1.,  0.,  1.],
                [ 1.,  1.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  0., 1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  1.],
                [ 1.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  1., 1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  1.],
                [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  1., 0.,  1.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  1.],
                [ 1.,  0.,  1.,  1.,  0.,  0.,  0.,  1.,  0.,  1., 0.,  1.,  1.,  0.,  1.,  1.,  1.,  0.,  0.,  1.],
                [ 1.,  0.,  1.,  1.,  0.,  1.,  1.,  1.,  0.,  1., 0.,  1.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  1.],
                [ 1.,  0.,  1.,  0.,  1.,  1.,  0.,  0.,  0.,  1., 0.,  1.,  1.,  1.,  1.,  0.,  1.,  0.,  1.,  1.],
                [ 1.,  0.,  1.,  0.,  1.,  0.,  0.,  1.,  1.,  1., 0.,  1.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,  1.],
                [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.,  0.,  0., 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]
            ])
        # Medium maze environment 10x10
        elif self.difficulty == 'Medium':
            self.env = np.array([
                [ 1.,  1.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,  1.],
                [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.],
                [ 1.,  1.,  0.,  1.,  1.,  0.,  1.,  0.,  0.,  1.],
                [ 0.,  0.,  0.,  1.,  1.,  1.,  0.,  1.,  1.,  1.],
                [ 1.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  1.,  1.],
                [ 1.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.],
                [ 1.,  1.,  1.,  0.,  0.,  0.,  0.,  1.,  1.,  0.],
                [ 1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.],
                [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
                [ 1.,  1.,  0.,  1.,  1.,  1.,  0.,  1.,  1.,  1.],
                [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.]
            ])
        # Easy maze environment 5x5
        elif self.difficulty == 'Easy':
            self.env = np.array([
                [ 1.,  1.,  0.,  1.,  1.],
                [ 0.,  1.,  0.,  1.,  0.],
                [ 1.,  1.,  1.,  1.,  1.],
                [ 1.,  0.,  0.,  1.,  0.],
                [ 1.,  1.,  1.,  1.,  1.]
            ])
        
        # Number of rows an columns
        self.n_rows, self.n_cols = np.shape(self.env)
        
        # Reset environment
        self.reset_env = np.copy(self.env)
        
        # Number of states
        self.n_states = np.sum(self.env == 1)
        
        # State tracking matrix for q-learning
        self.state_tracker = np.copy(self.env)
        self.state_counter = 0
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if self.state_tracker[i, j] == 1:
                    self.state_tracker[i,j] = self.state_counter
                    self.state_counter += 1
                else:
                    self.state_tracker[i, j] = -1
                    
        # Current state in state matrix and history
        self.state = 0
        self.state_history = []
        self.state_history.append(self.state)
        
        # Number of cells
        self.free_cells = self.n_states
        
        # Number of actions
        self.n_actions = 4
        # Actions: N, S, E, W
        self.action_dict = {
            0: (-1, 0),     # Never
            1: (0, 1),      # Eat
            2: (1, 0),      # Soggy
            3: (0, -1)}     # Waffles
        # Action term
        self.action_term = { 0: 'Up',
                    1: 'Right',
                    2: 'Down',
                    3: 'Left'}
        
        # Total elements
        self.size = self.env.size
        
        # Visualize greyscale mark 
        # 0 - black:invalid, 1 - white:valid
        self.visited_mark = 0.8
        self.mouse_mark = 0.3
        self.cheese_mark = 0.5
        
        # Position of cheese
        self.cheese_pos = (self.n_rows-1, self.n_cols-1)
        
        # Mouse position inititialized at start
        self.mouse_pos = (0, 0)
        self.prev_pos  = (0, 0)
        
        # Visited history
        self.visited_history = []
        
        # Action integer from 0-3
        self.action = -1
        #Track action history for debug
        self.action_history = []
        
        # Initializing maze with above positions in environment
        self.env[self.mouse_pos] = self.mouse_mark
        self.env[self.cheese_pos] = self.cheese_mark
        self.env[self.prev_pos] = self.visited_mark
        
        # For resetting environment
        self.reset_env[self.mouse_pos] = self.mouse_mark
        self.reset_env[self.cheese_pos] = self.cheese_mark
        self.reset_env[self.prev_pos] = self.visited_mark
        
        # Checking validity for validity function
        self.valid_location = self.reset_env == 1
        # Because there are markers on those positions, manually change valididty
        self.valid_location[self.mouse_pos] = True
        self.valid_location[self.cheese_pos] = True
        
        # Number of invalid moves
        self.n_invalid_moves = 0
        
        # Rewards
        self.reward = 0
        # Track reward history for debug
        self.reward_history = []
        self.reward_tracker = np.zeros((np.shape(self.env)))
        
        # Move counter
        self.n_moves = 0
        self.move_history = []
        self.explore_moves = 0
        self.exploit_moves = 0
        
        # Learning rate
        self.lr = 1.5
        
        # Initialize q-matrix (q-quality)
        self.q = np.zeros((self.n_states, self.n_actions))
        
        # After performance, see the optimal moves produced purely from q
        self.optimal_move = -1 * np.ones((self.n_rows, self.n_cols))
        
        # Discount balance in future and immediate rewards
        self.gamma = .9
        
        # Percent separation to explore or exploit
        self.epsilon = 0.7
        self.epsilon_history = []
        
        # Visualizing important data
        self.table = PrettyTable()
        self.table.field_names = ['Moves', 'Action', 'State', 'Reward']
        
        # Move stopping condition
        self.move_stopping_condition = False
        
        # Game condition
        self.winner = False
        
        # Number of wins
        self.n_wins = 0
        self.n_losses = 0
        
        # Number of epochs
        self.n_epochs = 0