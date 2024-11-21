
# import environment
# import agent
# import matplotlib.pyplot as plt
# import matplotlib.animation as anim
# from matplotlib.animation import PillowWriter
# import numpy as np
# import fnn
# import pickle

# with open('agent_1.pkl', 'rb') as f:
#     params = pickle.load(f)

# weights = params['weights']
# biases = params['biases']
# layers = params['layers']

# # Initialize the sound gradient
# sound = environment.SoundGradient(source=(50, 50), decay_factor=0.001)
# sound_grid = sound.generate_gradient()
# a = fnn.FNN(layers)

# a.weights = weights
# a.biases = biases

# # # Create an agent at position (40, 40)
# # agn = agent.Agent(20, 40, sound_grid, a)

# start_x = np.random.randint(0, 100) 
# start_y = np.random.randint(0, 100)
# agn = agent.Agent(start_x, start_y, sound_grid, a)



# # Simulate the agent's movement, storing the path for animation
# positions = []
# initial_position = agn.get_position()
# positions.append(initial_position)
# runs = 0
# while sound_grid[agn.x, agn.y] < 0.98 and runs < 1000:
#     agn.move()
#     positions.append(agn.get_position())
#     runs += 1

# # Unzip the positions into x and y coordinates
# x_positions, y_positions = zip(*positions)

# # Set up the plot for animation
# fig, ax = plt.subplots()

# # Plot the sound gradient as the background (use alpha to make it light)
# sound.plot_gradient(alpha=0.4)

# # Plot the start and end points
# start_point = plt.scatter(initial_position[1], initial_position[0], color='red', s=100, label='Start Position')
# end_point = plt.scatter(agn.get_position()[1], agn.get_position()[0], color='green', s=100, label='Source Reached')

# # Plot the path with an initial blue line and points
# path_line, = plt.plot([], [], marker='o', color='blue', markersize=4, linewidth=1, label='Agent Path')

# # Set axis limits to zoom in around the agent's path
# min_x, max_x = min(x_positions), max(x_positions)
# min_y, max_y = min(y_positions), max(y_positions)
# plt.xlim(min_y - 5, max_y + 5)
# plt.ylim(min_x - 5, max_x + 5)

# plt.legend()
# plt.title('Agent Movement Path (Animation)')

# # Update function for the animation
# def update(frame): 
#     if frame >= len(x_positions):
#         frame = len(x_positions) - 1  
#     # Update the path with the current positions up to the current frame
#     path_line.set_data(y_positions[:frame + 1], x_positions[:frame + 1])
#     return path_line,

# # Create the animation
# ani = anim.FuncAnimation(fig, update, frames=len(x_positions), interval=200, blit=True)

# # Show the animation
# plt.show()

# # Save the animation as a GIF
# # ani.save("agent_path_animation.gif", writer=PillowWriter(fps=30))

# # Create a second static plot for the full path
# fig_static, ax_static = plt.subplots()

# # Plot the sound gradient
# sound.plot_gradient(alpha=0.4)

# # Plot the start and end points
# plt.scatter(initial_position[1], initial_position[0], color='red', s=100, label='Start Position')
# plt.scatter(agn.get_position()[1], agn.get_position()[0], color='green', s=100, label='Source Reached')

# # Plot the full path
# plt.plot(y_positions, x_positions, marker='o', color='blue', markersize=4, linewidth=1, label='Agent Path')

# # Set axis limits to zoom in around the agent's path
# plt.xlim(min_y - 5, max_y + 5)
# plt.ylim(min_x - 5, max_x + 5)

# plt.legend()
# plt.title('Agent Full Movement Path (Static)')

# # Show the static plot
# plt.show()


import environment
import agent
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.animation import PillowWriter
import numpy as np
import fnn
import pickle

with open('agent_2.pkl', 'rb') as f:
    params = pickle.load(f)

weights = params['weights']
biases = params['biases']
layers = params['layers']

# Initialize the sound gradient
sound = environment.SoundGradient(source=(50, 50), decay_factor=0.001)
sound_grid = sound.generate_gradient()
a = fnn.FNN(layers)

a.weights = weights
a.biases= biases

# Create an agn at position (40, 40)
# agn = agent.Agent(30, 30, sound_grid, a)
# agn = agent.Agent(20, 80, sound_grid, a)
# agn = agent.Agent(80, 80, sound_grid, a)
# agn = agent.Agent(80, 20, sound_grid, a)

start_x = np.random.randint(0, 100) 
start_y = np.random.randint(0, 100)
agn = agent.Agent(start_x, start_y, sound_grid, a)


# Simulate the agn's movement, but store the path for animation
positions = []
initial_position = agn.get_position()
positions.append(initial_position)
runs = 0
while sound_grid[agn.x, agn.y] < 0.98 and runs < 1000:
    # agn.move_no_network()
    agn.move()
    positions.append(agn.get_position())
    runs += 1

# Unzip the positions into x and y coordinates
x_positions, y_positions = zip(*positions)

# Set up the plot for animation
fig, ax = plt.subplots()

# Plot the sound gradient as the background (use alpha to make it light)
sound.plot_gradient(alpha=0.4)

# Plot the start and end points
start_point = plt.scatter(initial_position[1], initial_position[0], color='red', s=100, label='Start Position')
end_point = plt.scatter(agn.get_position()[1], agn.get_position()[0], color='green', s=100, label='Source Reached')

# Plot the path with an initial blue line and points
path_line, = plt.plot([], [], marker='o', color='blue', markersize=4, linewidth=1, label='agn Path')

# Set axis limits to zoom in around the agn's path
min_x, max_x = min(x_positions), max(x_positions)
min_y, max_y = min(y_positions), max(y_positions)
plt.xlim(min_y - 5, max_y + 5)
plt.ylim(min_x - 5, max_x + 5)

plt.legend()
plt.title('Agent Movement Path')

# Update function for the animation
def update(frame): 
    if frame >= len(x_positions):
        frame = len(x_positions) - 1  
    # Update the path with the current positions up to the current frame
    path_line.set_data(y_positions[:frame + 1], x_positions[:frame + 1])
    return path_line,

# # Create the animation
# ani = anim.FuncAnimation(fig, update, frames=len(x_positions), interval=200, blit=True)


# # Show the animation
# plt.show()
# # ani.save("old_fitness_results.gif", writer='PillowWriter',fps=30)



# Create a second static plot for the full path
fig_static, ax_static = plt.subplots()

# Plot the sound gradient
sound.plot_gradient(alpha=0.4)

# Plot the start and end points
plt.scatter(initial_position[1], initial_position[0], color='red', s=100, label='Start Position')
plt.scatter(agn.get_position()[1], agn.get_position()[0], color='green', s=100, label='Source Reached')

# Plot the full path
plt.plot(y_positions, x_positions, marker='o', color='blue', markersize=4, linewidth=1, label='Agent Path')

# Set axis limits to zoom in around the agent's path
plt.xlim(min_y - 5, max_y + 5)
plt.ylim(min_x - 5, max_x + 5)

plt.legend()
plt.title('Agent Full Movement Path (Static)')

# Show the static plot
plt.show()