# import numpy as np
# import matplotlib.pyplot as plt
# import agent  
# import fnn
# import pickle
# import environment

# source_x, source_y = 50, 50 
# num_circles = 4
# agents_per_circle = 25
# circle_radii = [10, 20, 30, 40] 
# grid_size = 100
# all_agents = []


# with open('agent_1_test.pkl', 'rb') as f:
#     params = pickle.load(f)

# weights = params['weights']
# biases = params['biases']
# layers = params['layers']

# # Initialize the sound gradient
# sound = environment.SoundGradient(source=(50, 50), decay_factor=0.001)
# sound_grid = sound.generate_gradient()
# a = fnn.FNN(layers)

# a.weights = weights
# a.biases= biases

# for r in circle_radii:
#     for i in range(agents_per_circle):
        
#         angle = 2 * np.pi * i / agents_per_circle
       
#         x = int(round(source_x + r * np.cos(angle)))
#         y = int(round(source_y + r * np.sin(angle)))
        
#         x = max(0, min(grid_size - 1, x))
#         y = max(0, min(grid_size - 1, y))
      
#         agn = agent.Agent(x, y, sound_grid, a) 
#         all_agents.append(agn)


# plt.figure(figsize=(8, 8))
# plt.xlim(0, grid_size)
# plt.ylim(0, grid_size)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, color='gray')

# # Plot source
# plt.scatter(source_x, source_y, c='red', label='Source Point', s=100)

# # Plot agents
# for agn in all_agents:
#     plt.scatter(agn.x, agn.y, c='blue', s=10)

# # Labels and title
# plt.title("Agents Placed on Concentric Circles")
# plt.xlabel("X Coordinate")
# plt.ylabel("Y Coordinate")
# plt.legend()
# plt.show()


# success_radius = 1.5  # Distance considered a success
# max_steps = 1000  # Maximum steps each agent can take
# success_count = 0  # Counter for successful agents

# # Iterate through all agents
# for agn in all_agents:
#     # Store the agent's path for debugging or visualization
#     positions = []
#     initial_position = agn.get_position()
#     positions.append(initial_position)

#     runs = 0
#     while runs < max_steps:
#         agn.move()  
#         current_x, current_y = agn.get_position()
#         positions.append((current_x, current_y))

#         # Check if the agent is within the success radius of the source
#         if np.sqrt((current_x - source_x)**2 + (current_y - source_y)**2) <= success_radius:
#             success_count += 1
#             break  # Stop further movement for this agent if it succeeded

#         runs += 1

# # Display the results
# total_agents = len(all_agents)
# print(f"Number of successes: {success_count} out of {total_agents}")
# print(f"Success rate: {success_count / total_agents * 100:.2f}%")
import numpy as np
import matplotlib.pyplot as plt
import agent  
import fnn
import pickle
import environment

# Configuration for the test
source_x, source_y = 50, 50 
num_circles = 4
agents_per_circle = 25
circle_radii = [10, 20, 30, 40] 
grid_size = 100
all_agents = []

# Load the neural network parameters
with open('agent_16_test.pkl', 'rb') as f:
    params = pickle.load(f)

weights = params['weights']
biases = params['biases']
layers = params['layers']

# Initialize the sound gradient
sound = environment.SoundGradient(source=(50, 50), decay_factor=0.001)
sound_grid = sound.generate_gradient()
a = fnn.FNN(layers)
a.weights = weights
a.biases = biases

# Place agents on concentric circles
for r in circle_radii:
    for i in range(agents_per_circle):
        angle = 2 * np.pi * i / agents_per_circle
        x = int(round(source_x + r * np.cos(angle)))
        y = int(round(source_y + r * np.sin(angle)))
        x = max(0, min(grid_size - 1, x))
        y = max(0, min(grid_size - 1, y))
        agn = agent.Agent(x, y, sound_grid, a)
        all_agents.append(agn)

# Success metrics
success_radius = 1.5  # Distance considered a success
max_steps = 1000  # Maximum steps each agent can take
success_count = 0  # Counter for successful agents
successful_paths = []  # Store paths of successful agents

# Simulate agent movement
for agn in all_agents:
    positions = []  # Track positions
    initial_position = agn.get_position()
    positions.append(initial_position)

    runs = 0
    while runs < max_steps:
        agn.move()
        current_x, current_y = agn.get_position()
        positions.append((current_x, current_y))

        # Check success condition
        if np.sqrt((current_x - source_x)**2 + (current_y - source_y)**2) <= success_radius:
            success_count += 1
            successful_paths.append(positions)  # Save path of successful agent
            break

        runs += 1

# Display overall results
total_agents = len(all_agents)
print(f"Number of successes: {success_count} out of {total_agents}")
print(f"Success rate: {success_count / total_agents * 100:.2f}%")

# Plot paths of successful agents
plt.figure(figsize=(8, 8))
plt.xlim(0, grid_size)
plt.ylim(0, grid_size)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, color='gray')

# Plot source
plt.scatter(source_x, source_y, c='red', label='Source Point', s=100)

# Plot paths
for path in successful_paths:
    x_positions, y_positions = zip(*path)
    plt.plot(y_positions, x_positions, marker='o', markersize=2, linewidth=0.5, label="Successful Path")

# Labels and title
plt.title("Paths of Successful Agents")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
# plt.legend(loc='lower left')
plt.show()


# Data
num_agents = [1, 2, 4, 8, 12, 16]
success_percent = [27, 52, 52, 27, 53, 53]

# Plot configuration
plt.figure(figsize=(8, 6))
plt.plot(num_agents, success_percent, marker='o', linestyle='-', color='blue', label="Success Percentage")
plt.title("Number of Agents vs Success Percentage")
plt.xlabel("Number of Agents")
plt.ylabel("Success Percentage (%)")
plt.grid(visible=True, linestyle='--', linewidth=0.5, color='gray')
plt.xticks(num_agents)  # Ensure x-axis matches the agent numbers
plt.legend(loc="best")
plt.show()