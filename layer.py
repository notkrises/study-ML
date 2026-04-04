from typing import List
from neuron import calculateWeightedSum, sigmoid

inputs = [0.5, 0.8, 0.2]
weights = [[0.4, -0.3, 0.6], [0.1, 0.5, -0.2]]
biases = [-0.1, 0.3]


# Calculates the activations for each neuron in a layer using the inputs array, weights array, and biases array as a parameter
def layer(
    inputs: List[float], weights: List[List[float]], biases: List[float]
) -> List[float]:

    # setting up activations array and calculating number of neurons
    activations = []
    numNeurons = len(biases)

    # Calculates the activation for each neuron in the layer
    for index in range(numNeurons):
        # extracting weights and biases
        current_weights = weights[index]
        current_bias = biases[index]

        # calculating weighted sum and squished sum
        weighted_sum = calculateWeightedSum(inputs, current_weights, current_bias)
        squished = sigmoid(weighted_sum)

        # adds each neuron's activation to the activations array
        activations.append(squished)

    return activations


print(layer(inputs, weights, biases))
