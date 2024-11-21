import numpy as np
import matplotlib.pyplot as plt

class SoundGradient:
    def __init__(self, source, decay_factor=0.1, grid_size=100):
        self.source = source  # (x, y) of the sound source
        self.decay_factor = decay_factor  # How fast the sound intensity decays
        self.grid_size = grid_size  # Size of the 2D grid
        self.grid = np.zeros((grid_size, grid_size))  # Grid to store sound intensities
    
    def generate_gradient(self):
        max_value = 1.0
        min_value = 0.008
        drop_rate = 0.1 / 5  # Approximately 0.1 every 5 units

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                distance = np.sqrt((i - self.source[0]) ** 2 + (j - self.source[1]) ** 2)
                
                if distance == 0:
                    self.grid[i, j] = max_value
                else:
                    # The value should decrease slowly, dropping 0.1 every 5 units
                    value = max_value - (drop_rate * distance)
                    # Ensure the value doesn't go below the minimum value
                    self.grid[i, j] = max(min_value, value)
                
                # Round the values to two decimal places
                self.grid[i, j] = round(self.grid[i, j], 3)
        
        return self.grid

    # def generate_gradient(self):
    #     for i in range(self.grid_size):
    #         for j in range(self.grid_size):
    #             distance = np.sqrt((i - self.source[0]) ** 2 + (j - self.source[1]) ** 2)
                
    #             if distance == 0:
    #                 self.grid[i, j] = 1.0
    #             else:
    #                 self.grid[i, j] = 1 / (distance ** 2 + self.decay_factor)
    #     return self.grid
    
    def plot_gradient(self, alpha = 1.0):
        plt.imshow(self.grid, cmap='hot', interpolation='nearest', alpha=alpha, origin='lower')
        plt.colorbar(label='Sound Intensity')
        plt.title('2D Sound Gradient')

# # Create a sound source at position (50, 50)
# sound = SoundGradient(source=(50, 50), decay_factor = 1)
# sound_grid = sound.generate_gradient()
# sound.plot_gradient()
# plt.show()
# print(sound.grid)