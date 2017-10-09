import random
from collections import namedtuple

trial_run = namedtuple('trial_run', ['inputs', 'solution'])

class trainingArray:
    def __init__(self, n, examples):
        self.n = n
        self.examples = examples

    def createTrainingData(self):
        ''' Loop through the specified number of examples. For each
            iteration, generate n random inputs to the Rosenbrock function,
            calculate the output, store both in an array, return the array. '''

        trainingData = []

        for i in range(self.examples):
            trainingData.append(self.createInstance())   

        return trainingData        

    def createInstance(self):
        ''' Produce one randomized input and solution to the Rosenbrock
            function. Return an tuple of the array of inputs and the solution. '''

        functionInputs = []

        for i in range(self.n):
                functionInputs.append(random.uniform(-1, 1)) #update this based on bounds of function

        solution = self.solveRosenbrock(functionInputs)
        return trial_run(functionInputs, solution)

    def solveRosenbrock(self, inputs):
        ''' Given an array of inputs, return the solution to the
            Rosenbrock function. '''

        functionSum = 0

        for i in range(len(inputs) - 1):
            tempSum = ((1 - inputs[i]) ** 2) + (100 * ((inputs[i + 1] - (inputs[i] ** 2)) ** 2))
            functionSum += tempSum

        return functionSum