import numpy as np


class neuron:
    def __init__(self, function_type, sigma=None):
        self.function_type = function_type
        self.sigma = sigma

    def calculate_output(self, i_inputs=None, i_want_derivative=False, in_Kvectors=None):
        # Given a vector of inputs and an array of weights
        # use the activation function to calculate the output '''

        output = 0

        if self.function_type == "linear":
            output = i_inputs

        elif self.function_type == "sigmoid":
            if i_want_derivative:
                logit = 1 / (1 + np.exp(-i_inputs))
                output = logit * (1 - logit)
            else:
                output = 1 / (1 + np.exp(-i_inputs))

        elif self.function_type == "hyperbolic":
            if i_want_derivative:
                output = 1 - np.tanh(i_inputs)**2
            else:
                output = np.tanh(i_inputs)

        elif self.function_type == "gaussian":
            if i_want_derivative:  # Derivatives not needed for backprop in RBF network
                output = np.zeros(len(i_inputs))
            else:
                # in_Kvectors is a list of tuples
                num_nodes = len(in_Kvectors)
                output = np.zeros(num_nodes)
                for i in range(num_nodes):
                    output[i] = np.exp(-((np.linalg.norm(np.subtract(i_inputs, in_Kvectors[i])) ** 2) / (2 * (self.sigma ** 2))))

        return output
