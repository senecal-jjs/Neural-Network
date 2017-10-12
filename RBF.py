import numpy as np
from typing import Sequence
from trainingArray import trial_run
import Layer

'''
   The network class in RBF.py is used to initialize a radial basis function network. It contains the methods
   required to train the network, as well as a method to calculate the output of the network given an arbitrary number
   of inputs.

   Training of the network weights is done using incremental training

   The network can be created with an arbitrary number of nodes per layer.
'''

class network:
    # To create a RBF network provide the neurons desired for each layer, along with the activation function that
    # will be used in the hidden layer neurons (always a gaussian), and the centroids of the gaussian functions.
    def __init__(self, neurons_per_layer, activation_function, k_means_vectors):
        self.layers = []
        self.num_layers = len(neurons_per_layer)

        # The parameter 'sigma' required  by the guassian functions is calculated using the common
        # heuristic sigma = d/sqrt(2k), where d is the max distance between two clusters, and k is the number
        # of clusters
        self.sigma = self.calculate_sigma(k_means_vectors)

        # Create the three layers of the network, input, hidden, output
        self.layers.append(Layer.layer([neurons_per_layer[0] + 1, neurons_per_layer[0 + 1]], "linear", input_layer=True))
        self.layers.append(Layer.layer([neurons_per_layer[1] + 1, neurons_per_layer[1 + 1]], activation_function,
                                       in_sigma=self.sigma, k_means=k_means_vectors))
        self.layers.append(Layer.layer([neurons_per_layer[-1], None], "linear", output_layer=True))

        self.previous_weight_change = np.zeros(self.layers[1].weights.shape)

    # Given a set of inputs to the input layer, calculate the output of each layer in the network
    # and return the output of the final layer
    def calculate_outputs(self, inputs):
        self.layers[1].inputs = inputs
        self.layers[2].inputs = self.layers[1].calculate_output()
        return self.layers[-1].calculate_output()

    # Weights only exist between the hidden layer and the output layer so backpropagation error is only
    # calculated at the output layer
    def backpropagate(self, network_output, true_value):
        self.layers[-1].delta_values = (network_output - true_value)

    # Using the error that was calculated in the final layer of the network, calculate what the weight update
    # should be using gradient descent. Return the weight changes that are applied between the hidden and
    # output layer
    def update_weights(self, learning_rate, use_momentum=False, beta=None):
        if use_momentum:
            weight_change = -learning_rate * np.outer(self.layers[2].delta_values, self.layers[1].outputs).T
            self.layers[1].weights += (weight_change + beta * self.previous_weight_change)
            self.previous_weight_change = (weight_change + beta * self.previous_weight_change)
        else:
            weight_change = -learning_rate * np.outer(self.layers[2].delta_values, self.layers[1].outputs).T
            self.layers[1].weights += weight_change

    # The weights between the hidden and output layer are updated after every training example
    def train_incremental(self, training_data: Sequence[trial_run], learning_rate, use_momentum=False, beta=None):
        for data_point in training_data:
            output = self.calculate_outputs(data_point.inputs)
            self.backpropagate(output, data_point.solution)
            self.update_weights(learning_rate, use_momentum=use_momentum, beta=beta)


    # This method calculates the parameter 'sigma' which is required by the gaussian activation functions
    @staticmethod
    def calculate_sigma(k_means_vectors):
        max_distance = 0
        for vector_1 in k_means_vectors:
            for vector_2 in k_means_vectors:
                distance = np.linalg.norm(np.subtract(vector_1, vector_2))
                if distance > max_distance:
                    max_distance = distance

        sigma = max_distance/np.sqrt(2*len(k_means_vectors))
        return sigma
