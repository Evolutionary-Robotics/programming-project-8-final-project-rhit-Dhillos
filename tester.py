import numpy as np

dataset = [] 
labels = []

def generate_trainingData(N) : 
  dataset = []
  labels = []
  for i in range(N) : 
    input = np.zeros(4) 
    label = np.zeros(4) 
    input = np.random.rand(4)
    label[np.argmax(input)] = 1
    dataset.append(input) 
    labels.append(label)
  return dataset, labels

def distance(x1, x2, y1, y2): 
  return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

print(distance(0, 3, 0, 4))
# dataset, labels = generate_trainingData(100)
# print(f"dataset : {dataset} \n labels : {labels}")