import numpy as np
import matplotlib.pyplot as plt
from Maze_Class import Maze_Class
import random
from prettytable import PrettyTable
import time
from scipy.optimize import curve_fit
plt.close('all')

# Assign maze to all variables in Maze Class
maze = Maze_Class()

# Visualization of the maze
def show(maze):
    plt.figure()
    plt.grid('on')
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, maze.n_rows, 1))
    ax.set_yticks(np.arange(0.5, maze.n_cols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.title('Mouse-Maze RL - Difficulty: ' + maze.difficulty)
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.imshow(maze.env, cmap='gray')

    plt.show()

# Plot epochs vs. number of moves
def plot_results(maze):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(0, maze.n_epochs), maze.move_history)
    plt.title('Mouse-Maze RL - Epochs vs. Moves')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Number of Moves')
    
    plt.show()

# Show on path, the direction of what the agent learned from q-table
def arrow(maze):
    direction = []
    arrow_actions = {
            0: (0, 1),      # North
            1: (1, 0),      # East
            2: (0, -1),     # South
            3: (-1, 0)}     # Wests
    for idx in range(len(maze.q)):
        action = np.argmax(maze.q[idx, :])
        direction.append(arrow_actions[action])
        
    plt.figure()
    ax = plt.gca()
    plt.grid('on')
    ax.set_xticks(np.arange(0.5, maze.n_rows, 1))
    ax.set_yticks(np.arange(0.5, maze.n_cols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.imshow(maze.env, cmap='gray')
    state = 0
    for i in range(maze.n_rows):
        for j in range(maze.n_cols):
            if maze.state_tracker[j, i] != -1:
                x_direct = direction[state][0]
                y_direct = direction[state][1]
                ax.quiver(i, j, x_direct, y_direct)
                state += 1
    plt.show()

# Plot maze with state number over each state
def plot_state_matrix(maze):
    plt.figure()
    plt.grid('on')
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, maze.n_rows, 1))
    ax.set_yticks(np.arange(0.5, maze.n_cols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.title('Mouse-Maze RL - States for Difficulty: ' + maze.difficulty)
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    reset(maze)
    plt.imshow(maze.env, cmap='gray')
    state = 0
    for i in range(maze.n_rows):
        for j in range(maze.n_cols):
            if maze.state_tracker[i,j] != -1:
                ax.annotate(str(state), xy=(j, i),xytext=(j,i))
                state += 1
    plt.show()
# Plot on plot_results function the regression trend line of number of moves vs. number of games played
# 3rd degree function to model regression line, line of best fit will vary
def func(x, a, b, c, d):
    return a*(x**3) + b*(x**2) + c*x + d
def convergence(maze):
    # Games played list
    games_played = np.arange(0, maze.n_epochs)
    # Call results plot move history vs games played
    plot_results(maze)
    # Fit regression line
    popt, pcov = curve_fit(func, games_played, maze.move_history)
    # Plot regression line
    plt.plot(games_played, func(games_played, *popt), 'r-',label='fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f' % tuple(popt))
    
    plt.show()
    
    
# Completes an action, checks action validity, and gets new reward and state
def act(maze, action):
    status_check = 'invalid'
    # Moving the mouse
    new_state = np.add(maze.mouse_pos, maze.action_dict[action])
    
    # Check if move is possible
    new_move = (new_state[0], new_state[1])
    # Call valid move function
    status_check = valid_move(maze, new_move)
    
    if status_check == 'valid':
        # Update visited position history
        maze.prev_pos = maze.mouse_pos
        # Updates visited and doesn't include if been there x many times
        if maze.prev_pos not in maze.visited_history:
            maze.visited_history.append(maze.prev_pos)
        
        # Assign new state to mouse position
        maze.mouse_pos = (new_move)
        
        # Update action history
        maze.action_history.append(action)
        
        # Update maze environment mark
        maze.env[maze.mouse_pos] = maze.mouse_mark
        maze.env[maze.prev_pos] = maze.visited_mark
        
        # Update maze current state
        get_state(maze)
    
    # Update number of moves
    maze.n_moves += 1
    
    # Obtain reward or penalty
    maze.reward = reward(maze, status_check)
    maze.reward_history.append(maze.reward)
    
# Rewards
def reward(maze, status_check):
    # Finding the cheese!
    if maze.mouse_pos == maze.cheese_pos:
        reward = 5
        maze.reward_tracker[maze.mouse_pos] = maze.reward_tracker[maze.mouse_pos] + reward
        return reward
    
    # If agent was just in that state, penalize for bad move, learn to make different moves
    if len(maze.state_history) > 3:
        if maze.state_history[len(maze.state_history)-3] == maze.state_history[-1]:
            reward = -2
            maze.reward_tracker[maze.mouse_pos] = maze.reward_tracker[maze.mouse_pos] + reward
            return reward
            
    # Visting a previous position
    if maze.mouse_pos in maze.visited_history:
        reward = -.3
        maze.reward_tracker[maze.mouse_pos] = maze.reward_tracker[maze.mouse_pos] + reward
        return reward
    
    # Invalid move penalty
    if status_check == 'invalid':
        reward = -3
    # Move into a free cell reward
    else:
        reward = 0.4
    
    maze.reward_tracker[maze.mouse_pos] = maze.reward_tracker[maze.mouse_pos] + reward
    
    return reward
    
# Did the mouse win or lose
def game_status(maze):
    if maze.mouse_pos == maze.cheese_pos:
        return 'winner'
    else:
        return 'loser'

# Reset the environment to initial parameters
def reset(maze):
    maze.action_history = []
    maze.visited_history = []
    maze.mouse_pos = (0, 0)
    maze.prev_pos = (0, 0)
    maze.state = 0
    maze.state_history = []
    maze.state_history.append(maze.state)
    maze.env = np.copy(maze.reset_env)
    maze.env[maze.mouse_pos] = maze.mouse_mark
    maze.env[maze.cheese_pos] = maze.cheese_mark
    maze.env[maze.prev_pos] = maze.visited_mark
    maze.n_moves = 0
    maze.n_invalid_moves = 0
    maze.reward_history = []
    maze.table = PrettyTable()
    maze.table.field_names = ['Moves', 'Action', 'State', 'Reward']
    
# Check if action is possible
def valid_move(maze, new_move):
    if new_move == maze.cheese_pos:
        return 'valid'
    elif (new_move[1] == -1) | (new_move[0] == -1):
        maze.n_invalid_moves += 1
        return 'invalid'
    elif (new_move[0] == maze.n_rows) | (new_move[1] == maze.n_cols):
        maze.n_invalid_moves += 1
        return 'invalid'
    elif maze.valid_location[new_move] == True:
        return 'valid'
    else:
        maze.n_invalid_moves += 1
        return 'invalid'

# Retrieve state of the mouse
def get_state(maze):
    maze.state = int(maze.state_tracker[maze.mouse_pos[0], maze.mouse_pos[1]])
    maze.state_history.append(int(maze.state))
    
# Randomly roam around the maze
def explore(maze):
    maze.action = random.randint(0, 3)
    
    act(maze, maze.action)
    if maze.n_moves > 2:
        maze.q[maze.state_history[len(maze.state_history)-2], maze.action] = \
            maze.q[maze.state_history[len(maze.state_history)-2], maze.action] + maze.lr * \
            (maze.reward + maze.gamma * np.max(maze.q[maze.state, :]) \
            - maze.q[maze.state_history[len(maze.state_history)-2], maze.action])
    # Obtain important maze information
    maze_info(maze)

# Utilize rewards to determine optimal next move
def exploit(maze):
    #Q[state, action] = Q[state, action] + 
    #                   lr * (reward + gamma * np.max(Q[new_state, :]) 
    #                   - Q[state, action])
    if maze.n_moves > 2:
        maze.q[maze.state_history[len(maze.state_history)-2], maze.action] = \
            maze.q[maze.state_history[len(maze.state_history)-2], maze.action] + maze.lr * \
            (maze.reward + maze.gamma * np.max(maze.q[maze.state, :]) \
            - maze.q[maze.state_history[len(maze.state_history)-2], maze.action])
        
    # Obtain optimal action
    maze.action = np.argmax(maze.q[maze.state, :])

    # Conduct the action
    act(maze, maze.action)
    
    # Update table
    maze_info(maze)
    
# Print important information about the maze
def maze_info(maze):    
    maze.table.add_row([maze.n_moves, maze.action_term[maze.action], 
                        maze.state, maze.reward])

# Maze move stopping condition
def game_difficulty_stopping_condition(maze):
    if maze.difficulty == 'Easy':
        if maze.n_moves <= 9:
            maze.move_stopping_condition = True
    if maze.difficulty == 'Medium':
        if maze.n_moves <= 26:
            maze.move_stopping_condition = True
    if maze.difficulty == 'Hard':
        if maze.n_moves <= 55:
            maze.move_stopping_condition = True

# Increase epsilon and lr if surpassed move limits
def game_difficulty_update_epsilon_lr(maze):
    # Easy difficulty
    if maze.difficulty == 'Easy':
        # Increase epsilon if exceeds certain number of moves

        if maze.n_moves == 1000:
            maze.epsilon += .025
            maze.lr += .075
  
        # Cutoff number of moves as a loss and continue next episode
        if maze.n_moves == 10000:
            maze.epsilon += .1
            maze.lr += .075
            maze.winner = True
            maze.n_losses += 1
        
    # Medium difficulty    
    if maze.difficulty == 'Medium':
        # Increase epsilon if exceeds certain number of moves
        if maze.n_moves == 10000:
            maze.epsilon += .025
            maze.lr += .075
            
        # Cutoff number of moves as a loss and continue next episode
        if maze.n_moves == 50000:
            maze.epsilon += .1
            maze.lr += .075
            maze.winner = True
            maze.n_losses += 1
          
    # Hard difficulty
    if maze.difficulty == 'Hard':
       # Increase epsilon if exceeds certain number of moves
        #if maze.n_moves == 50000:
            #maze.epsilon += .025
            #maze.lr += .075
            
        # Cutoff number of moves as a loss and continue next episode
        if maze.n_moves == 5000:
            #maze.epsilon += .1
            #maze.lr += .075
            maze.winner = True
            maze.n_losses += 1
            
# Normalize epsilon and learning rate
def normalize_epsilon_lr(maze):
    if maze.difficulty == 'Easy':
        if maze.epsilon > .2:
            maze.epsilon -= .025
    elif maze.difficulty == 'Medium':
        if maze.epsilon > .25:
            maze.epsilon -= .025
    elif maze.difficulty == 'Hard':
        if maze.epsilon > .2:
            maze.epsilon -= .025
    # Normalize lr
    if maze.lr > 1.2:
        maze.lr -= .1
        
# Main algorithm that loops learning process and calls other functions
def run_game(maze):
    # Begin timer
    start_time = time.time()
    
    print('Starting up the game')
    
    # Stoping condiion checker
    maze.move_stopping_condition = False
    
    while maze.move_stopping_condition == False:
        # Reset condition for new game
        reset(maze)
        # Game winning condition
        maze.winner = False
        while maze.winner == False:
            # Randomly explores the maze
            if random.uniform(0, 1) < maze.epsilon:
                explore(maze)
                maze.explore_moves += 1
            # Utilize q-table to determine optimal move 
            else:
                exploit(maze)
                maze.exploit_moves += 1
            
            # Check if game is a win, else continue
            if game_status(maze) == 'winner':
                maze.winner = True
                
            # Update epsilon and lr based off maze difficulty
            game_difficulty_update_epsilon_lr(maze)
            
        # Normalize epsilon and lr based off maze difficulty after each epoch
        normalize_epsilon_lr(maze)
        
        # Store epsilon change over number of epochs
        maze.epsilon_history.append(maze.epsilon)
        # Store move history over number of epochs
        maze.move_history.append(maze.n_moves)
        
        # Checks stopping condition based off maze difficulty
        game_difficulty_stopping_condition(maze)
        
        # Print result every epoch
        print('Epoch:', maze.n_epochs+1)
        print('Moves:', maze.n_moves)
        print('Epsilon {:0.4f}'.format(maze.epsilon))
        print('Learning Rate {:0.1f}'.format(maze.lr))
        print('__________________\n')
        maze.n_epochs += 1
        
    # Game completed, plot the results
    # Epochs vs. Moves
    plot_results(maze)
    
    # Time elapsed from initial run
    elapsed_time = time.time() - start_time
    
    # Print results after final optimal minimum moves achieved
    print('Elapsed time: {:.3f} seconds'.format(elapsed_time))
    print('Total epochs:', maze.n_epochs)
    print('Total wins:', maze.n_wins)
    print('Total losses:', maze.n_losses)
    print('Optimal move count:', maze.n_moves)

# Run the game
if __name__ == "__main__":
    run_game(maze)
    show(maze)
    arrow(maze)
    convergence(maze)