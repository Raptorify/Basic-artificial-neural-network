import numpy as np

# Input layer consisting of 4 neurons
inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

# Weight sets of 3 neurons inside hidden layer
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

# Bias of the 3 neurons inside the hidden layer
biases = [2.0, 3.0, 0.5]

# Second layer
weights2 = [[0.1, -0.14, 0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]

biases2 = [-1.0, 2.0, -0.5]

# Caclulates the dot product of the weight set and inputs and adds the neuron bias (loops through each weightset)
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
# Second layer input is the output from the first layer
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

# Print each neurons value(This value will be used for the activation function)
print(layer2_outputs)