import numpy as np
import matplotlib.pyplot as plt
import fnn 
import ea
import environment
import agent
import matplotlib.animation as anim
import pickle

def get_distance(x1, x2, y1, y2): 
  return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# Parameters of the neural network
layers = [4, 4]

# Parameters of the evolutionary algorithm
genesize = np.sum(np.multiply(layers[1:],layers[:-1])) + np.sum(layers[1:]) 
print("Number of parameters:",genesize)
popsize = 100 
recombProb = 0.5
mutatProb = 0.05
tournaments = 100*popsize 

sound = environment.SoundGradient(source=(50, 50), decay_factor=0.001)
sound_grid = sound.generate_gradient()


# def fitnessFunction(genotype, sound_grid, x, y):
#     snd_src_x = sound.source[0]
#     snd_src_y = sound.source[1]
    
#     a = fnn.FNN(layers)
#     a.setParams(genotype)
    
#     alist = []     
#     alist.append(agent.Agent(30, 30, sound_grid, a))
#     # alist.append(agent.Agent(70, 30, sound_grid, a))
#     # alist.append(agent.Agent(30, 70, sound_grid, a))
#     # alist.append(agent.Agent(70, 70, sound_grid, a)) 

#     # alist.append(agent.Agent(20, 20, sound_grid, a))
#     # alist.append(agent.Agent(20, 80, sound_grid, a))
#     # alist.append(agent.Agent(80, 80, sound_grid, a))
#     # alist.append(agent.Agent(80, 20, sound_grid, a)) 


#     # alist.append(agent.Agent(40, 60, sound_grid, a))
#     # alist.append(agent.Agent(60, 40, sound_grid, a))
#     # alist.append(agent.Agent(40, 40, sound_grid, a))
#     # alist.append(agent.Agent(60, 60, sound_grid, a))

#     # alist.append(agent.Agent(10, 10, sound_grid, a))
#     # alist.append(agent.Agent(90, 10, sound_grid, a))
#     # alist.append(agent.Agent(10, 90, sound_grid, a))
#     # alist.append(agent.Agent(90, 90, sound_grid, a))
    
    
    
#     total_distance_travelled = 0
#     max_distance = np.sqrt(sound_grid.shape[0]**2 + sound_grid.shape[1]**2)

#     agent_fitness = 0 
#     for agn in alist:    
#         runs = 0 
#         agn_x, agn_y = agn.get_position()
#         # print(agn_x, agn_y) 

#         while sound_grid[agn.x, agn.y] < 0.97 and runs < 100:
#             agn_x, agn_y = agn.get_position()
#             # step_distance = get_distance(snd_src_x, agn_x, snd_src_y, agn_y)
#             # path_distance += step_distance  # Accumulate distance for each step
           
#             agn.move()  # Move the agent to the next position
#             runs += 1
        

#         agn_x, agn_y = agn.get_position()
#         final_distance = get_distance(snd_src_x, agn_x, snd_src_y, agn_y)
#         # total_distance_travelled += path_distance + final_distance
#         total_distance_travelled = final_distance

#         normalized_fitness = 1 - (total_distance_travelled/ (max_distance ))
#         # normalized_fitness = 1 - 1/total_distance_travelled 
        
#         normalized_fitness = max(0, min(1, normalized_fitness))
#         if final_distance > np.sqrt(2) : 
#             normalized_fitness = normalized_fitness / 2
#         else :
#             normalized_fitness = normalized_fitness * 2
#         agent_fitness += normalized_fitness 

#     fitness = agent_fitness / len(alist)
#     return fitness 

def fitnessFunction(genotype, sound_grid, x, y):
    snd_src_x, snd_src_y = sound.source  # Source position
    
    # Initialize the FNN and assign parameters
    a = fnn.FNN(layers)
    a.setParams(genotype)
    
    # Create agents at predefined starting positions
    alist = [
        agent.Agent(30, 30, sound_grid, a),
        agent.Agent(70, 30, sound_grid, a),
        agent.Agent(30, 70, sound_grid, a),
        agent.Agent(70, 70, sound_grid, a),

        agent.Agent(20, 20, sound_grid, a),
        agent.Agent(20, 80, sound_grid, a),
        agent.Agent(80, 80, sound_grid, a),
        agent.Agent(80, 20, sound_grid, a),

        agent.Agent(40, 60, sound_grid, a),
        agent.Agent(60, 40, sound_grid, a),
        agent.Agent(40, 40, sound_grid, a),
        agent.Agent(60, 60, sound_grid, a),

        agent.Agent(10, 10, sound_grid, a),
        agent.Agent(90, 10, sound_grid, a),
        agent.Agent(10, 90, sound_grid, a),
        agent.Agent(90, 90, sound_grid, a),


    ]
    
    max_distance = np.sqrt(sound_grid.shape[0]**2 + sound_grid.shape[1]**2)
    agent_fitness = 0 

    for agn in alist:    
        runs = 0 
        agn_x, agn_y = agn.get_position()
        start_distance = get_distance(snd_src_x, agn_x, snd_src_y, agn_y)
        progress_towards_source = 0
        total_distance_reduction = 0
        previous_distance = start_distance

        # Simulate agent movement
        while sound_grid[agn.x, agn.y] < 0.97 and runs < 100:
            agn.move()  
            runs += 1
            agn_x, agn_y = agn.get_position()
            current_distance = get_distance(snd_src_x, agn_x, snd_src_y, agn_y)
            
            # Check progress towards the source
            if current_distance < previous_distance:
                progress_towards_source += 1
            total_distance_reduction += (previous_distance - current_distance)
            previous_distance = current_distance

        # Final distance from the source
        final_distance = get_distance(snd_src_x, agn.x, snd_src_y, agn.y)

        # Normalize fitness components
        progress_score = progress_towards_source / runs if runs > 0 else 0
        reduction_score = total_distance_reduction / max_distance
        final_distance_score = 1 - (final_distance / max_distance)
        
        # Bonus for reaching the source
        if final_distance <= np.sqrt(2):
            final_distance_score *= 2
            reduction_score *= 1.5

        # Penalize oscillation or minimal movement
        if progress_towards_source < runs * 0.9:  # Less than 90% moves towards source
            progress_score *= 0.5

        # Aggregate normalized scores for the agent
        agent_fitness += (0.4 * final_distance_score + 0.4 * reduction_score + 0.2 * progress_score)

    # Average fitness across all agents
    fitness = agent_fitness / len(alist)
    return fitness

pop = np.random.random((popsize,genesize))*2 - 1 

# Evolve
# ga = ea.MGA_Elitism(fitnessFunction, sound_grid, genesize, popsize, recombProb, mutatProb, tournaments, pop, 0, 0) 
ga = ea.MGA(fitnessFunction, sound_grid, genesize, popsize, recombProb, mutatProb, tournaments, pop, 0, 0) 
ga.run()
ga.showFitness()

# Obtain best agent
best = int(ga.bestind[-1])
print(best)
a = fnn.FNN(layers)
a.setParams(ga.pop[best])

with open('agent_16_test.pkl', 'wb') as f:
    pickle.dump({'layers': layers,'weights': a.weights, 'biases': a.biases}, f)
