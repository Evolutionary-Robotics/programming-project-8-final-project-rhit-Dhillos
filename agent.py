import matplotlib.pyplot as plt 
import numpy as np
import random

class Agent:
    def __init__(self, x, y, sound_grid, network):
        self.x = x
        self.y = y
        self.sound_grid = sound_grid  # The sound intensity map
        self.grid_size = sound_grid.shape[0]
        self.network = network;
    
    def move_no_network(self):
        # List of possible movements: left, right, up, down
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        # Get sound intensity for current position
        current_sound = self.sound_grid[self.x, self.y]
        
        # Calculate the probability of moving towards each neighboring cell
        probabilities = [] 
        for move in moves:
            new_x = np.clip(self.x + move[0], 0, self.grid_size - 1)
            new_y = np.clip(self.y + move[1], 0, self.grid_size - 1)
            sound_intensity = self.sound_grid[new_x, new_y]
            probabilities.append(sound_intensity)
          
        
        total_intensity = sum(probabilities)
        probabilities = [p / total_intensity for p in probabilities]
        # chosen_move = random.choices(moves, probabilities)[0]
        chosen_move = moves[np.argmax(probabilities)]
        
        self.x = np.clip(self.x + chosen_move[0], 0, self.grid_size - 1)
        self.y = np.clip(self.y + chosen_move[1], 0, self.grid_size - 1)
    
    def move(self):
        # List of possible movements: left, right, up, down
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        # Get sound intensity for current position
        current_sound = self.sound_grid[self.x, self.y]
        
        # Calculate the probability of moving towards each neighboring cell
        probabilities = []
        for move in moves:
            new_x = np.clip(self.x + move[0], 0, self.grid_size - 1)
            new_y = np.clip(self.y + move[1], 0, self.grid_size - 1)
            sound_intensity = self.sound_grid[new_x, new_y]
            probabilities.append(sound_intensity)
        
        total_intensity = sum(probabilities)
        probabilities = [p / total_intensity for p in probabilities]
        
        # print(probabilities) 
        # print(self.network.forward(probabilities))

        chosen_move = moves[np.argmax(self.network.forward(probabilities))]   
        self.x = np.clip(self.x + chosen_move[0], 0, self.grid_size - 1)
        self.y = np.clip(self.y + chosen_move[1], 0, self.grid_size - 1)
    
    def get_position(self):
        return self.x, self.y

