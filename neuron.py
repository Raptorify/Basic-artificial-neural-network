import numpy as np

# Input layer consisting of 4 neurons
inputs = [1.0, 2.0, 3.0, 2.5]

# Weight sets of 3 neurons inside hidden layer
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

# Bias of the 3 neurons inside the hidden layer
biases = [2.0, 3.0, 0.5]

# Caclulates the dot product of the weight set and inputs and adds the neuron bias (loops through each weightset)
# np.dot() first argument defines the matrix shape
outputs = np.dot(weights, inputs) + biases

# Print each neurons value(This value will be used for the activation function)
print(outputs)