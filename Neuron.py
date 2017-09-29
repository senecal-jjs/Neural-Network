import ActivationFunction
import numpy as np

class neuron:
    def __init__(self, functionType, sigma = None):
        self.functionType = functionType
        self.sigma = sigma
     
    def calcOutput(inputs, weights, lastLayer):
    	''' Given a vector of inputs and a vector of weights,
			use the activation function to calculate the output '''
		actFunc = ActivationFunction.activationFunction(inputs, weights, self.sigma)

		output = 0

		#Last layer is always linear weighted sum
		if lastLayer:
			output = actFunc.weightedSum()

		elif self.functionType == "sigmoid":
			output = actFunc.sigmoid()

		elif self.functionType == "hyperbolic":
			output = actFunc.hyperTan()

		elif self.functionType == "gaussian":
			output = actFunc.gaussian()

		return output