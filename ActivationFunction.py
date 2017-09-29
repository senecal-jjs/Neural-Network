import numpy as np

class activationFunction:
	def __init__(self, inputs, weights, sigma = None):
		#self.weightedSum = np.sum(np.dot(inputs, weights))
		self.inputs = inputs
		self.weights = weights
		self.sigma = sigma

	def sigmoid(self):
		''' Implementation of the sigmoidal activation function '''
		return 1 / (1 + np.exp(-self.weightedSum()))

	def hyperTan(self):
		''' Implementation of the hyperbolic tangent activation function '''
		return np.tanh(self.weightedSum())

	def gaussian(self):
		''' Implementation of the gaussian basis function 
			Here, weights is the center value vector '''
		return np.exp(-((np.linalg.norm(np.subtract(self.inputs, self.weights)) ** 2) / (2 * (self.sigma ** 2))))

	def weightedSum(self):
		return np.sum(np.dot(self.inputs, self.weights))