import numpy as np

np.random.seed(0)

# Input layer consisting of 4 neurons
X = [[1.0, 2.0, 3.0, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons): # Pass the number of inputs and number of neurons from the following layer
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # Consider the array shape so that there is no need for a transpose
        self.biases = np.zeros((1, n_neurons)) # Create an array filled with zeros and shape 1 by number of neurons
    def forward(self, inputs): # Feeds the input forward through the neurons
        self.output = np.dot(inputs, self.weights) + self.biases

# Create layer objects
layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
layer2.forward(layer1.output)

print(layer2.output)
