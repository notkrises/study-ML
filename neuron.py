import math
from typing import List

inputs = [0.5, 0.8, 0.2]
weights = [0.4, -0.3, 0.6]
bias = -0.1


# Calculates weighted sum using the inputs array, weights array, and bias as parameters
def calculate_weighted_sum(
    inputs: List[float], weights: List[float], bias: float
) -> float:
    weightedSum = 0
    for index in range(len(inputs)):
        # Extracts the activation and weight from the arrays
        activation = inputs[index]
        weight = weights[index]

        # Calculate the weighted sum by summing each activation multiplied by its weight
        weighted = activation * weight
        weightedSum += weighted
    return weightedSum + bias


# Takes the weighted sum as a parameter and squishes it down to a number between 0-1 using the sigmoid function
def sigmoid(weighted_sum) -> float:
    # Exponentiate e^(-weighted_sum) as part of the sigmoid function; e is used because it makes the calculus easier and is most natural in the function
    exp = math.exp(-weighted_sum)
    return 1 / (1 + exp)


weighted_sum = calculate_weighted_sum(inputs, weights, bias)
squished_weighted_sum = sigmoid(weighted_sum)

print(squished_weighted_sum)
