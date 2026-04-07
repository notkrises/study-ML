from layer import layer
from typing import List

inputs = [0.5, 0.8]
all_weights = [[[0.4, -0.3], [0.1, 0.5]], [[0.6, 0.2], [-0.4, 0.8]]]
all_biases = [[-0.1, 0.3], [0.0, -0.2]]

# Network that calculates the output for multiple layers of a network using the activations array, weights array, and biases array
def network(
    activations: List[float], weights: List[List[List[float]]], biases: List[List[float]]
):
    
    # uses a for loop and layer function to calculate the activations for each layer, then returns the activations of the last layer(output layer)
    for index in range(len(biases)):
        activations = layer(activations, weights[index], biases[index])
    return activations


print(network(inputs, all_weights, all_biases))
