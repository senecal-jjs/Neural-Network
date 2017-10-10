import numpy as np
import Layer


class network:
    # include number of inputs as the first value in neurons_per_layer
    def __init__(self, neurons_per_layer, activation_function, k_means_vectors):
        self.layers = []
        self.num_layers = len(neurons_per_layer)

        # Create the three layers of the network, input, hidden, output
        self.layers.append(Layer.layer([neurons_per_layer[0] + 1, neurons_per_layer[0 + 1]], "linear", input_layer=True))
        self.layers.append(Layer.layer([neurons_per_layer[1] + 1, neurons_per_layer[1 + 1]], activation_function,
                                       in_sigma=0.1, k_means=k_means_vectors))
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

    def train_incremental(self):
        print('test')
