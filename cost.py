from typing import List

output = [0.1, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
expected = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


# Cost function that calculates the total cost using an output expected and expected array
def cost(output: List[float], expected: List[float]):
    # declaring and initializing the output of this function
    sum = 0

    # for loop that calculates the cost value of each output then adds it to the sum
    for index in range(len(output)):
        single_cost = (output[index] - expected[index]) ** 2
        sum += single_cost

    return sum


print(cost(output, expected))
