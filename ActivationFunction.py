import numpy as np

class activationFunction:
    def __init__(self, inputs=None, want_derivative = False, sigma = None, k_vectors = None):
        self.inputs = inputs
        self.sigma = sigma
        self.want_derivative = want_derivative
        self.k_vectors = k_vectors

    def sigmoid(self):
        ''' Implementation of the sigmoidal activation function '''
        if self.want_derivative:
            logit = 1 / (1 + np.exp(-self.inputs))
            return logit*(1-logit)
        else:
            return 1 / (1 + np.exp(-self.inputs))

    def hyperTan(self):
        ''' Implementation of the hyperbolic tangent activation function '''
        return np.tanh(self.inputs)

    def gaussian(self):
        ''' Implementation of the gaussian basis function
            Here, weights is the center value vector '''
        return np.exp(-((np.linalg.norm(np.subtract(self.inputs, self.k_vectors)) ** 2) / (2 * (self.sigma ** 2))))

    def weightedSum(self):
        return self.inputs
