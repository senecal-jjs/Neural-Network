import numpy as np
from typing import Sequence
from trainingArray import trial_run
import Layer


class network:
    # include number of inputs as the first value in neurons_per_layer
    def __init__(self, neurons_per_layer, activation_function, k_means_vectors):
        self.layers = []
        self.num_layers = len(neurons_per_layer)
        self.sigma = self.calculate_sigma(k_means_vectors)

        # Create the three layers of the network, input, hidden, output
        self.layers.append(Layer.layer([neurons_per_layer[0] + 1, neurons_per_layer[0 + 1]], "linear", input_layer=True))
        self.layers.append(Layer.layer([neurons_per_layer[1] + 1, neurons_per_layer[1 + 1]], activation_function,
                                       in_sigma=self.sigma, k_means=k_means_vectors))
        self.layers.append(Layer.layer([neurons_per_layer[-1], None], "linear", output_layer=True))

    # inputs is a one dimensional array containing the inputs to the function
    def calculate_outputs(self, inputs):
        self.layers[1].inputs = inputs
        self.layers[2].inputs = self.layers[1].calculate_output()
        return self.layers[-1].calculate_output()

    # Backpropgate on the last layer in the RBF network
    def backpropagate(self, network_output, true_value):
        self.layers[-1].delta_values = (network_output - true_value)

    def update_weights(self, learning_rate):
        weight_change = -learning_rate * np.outer(self.layers[2].delta_values, self.layers[1].outputs).T
        self.layers[1].weights += weight_change

    def train_incremental(self, training_data: Sequence[trial_run], learning_rate):
        for data_point in training_data:
            output = self.calculate_outputs(data_point.inputs)
            self.backpropagate(output, data_point.solution)
            self.update_weights(learning_rate)

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
