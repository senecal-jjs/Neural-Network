import numpy as np
import Layer


class network:
    # include number of inputs as the first value in neurons_per_layer
    def __init__(self, neurons_per_layer, activation_function, k_means_vectors):
        self.layers = []
        self.num_layers = len(neurons_per_layer)
        self.k_means = k_means_vectors

        for i in range(self.num_layers-1):
            # Create Input layer, the +1 in neurons_per_layer[i]+1 is to hold a bias value
            if i == 0:
                self.layers.append(Layer.layer([neurons_per_layer[i] + 1, neurons_per_layer[i + 1]], "linear", input_layer=True))

            # Create Hidden Layers
            else:
                self.layers.append(Layer.layer([neurons_per_layer[i] + 1, neurons_per_layer[i + 1]], activation_function))

        # Activate output layer
        self.layers.append(Layer.layer([neurons_per_layer[-1], None], "linear", output_layer=True))

    # inputs is a one dimensional array containing the inputs to the function
    def calculate_outputs(self, inputs, k_means_vectors):
        self.layers[0].outputs = np.append(inputs, 1)  # the 1 is added as a bias value

        for i in range(self.num_layers - 1):
            self.layers[i+1].inputs = self.layers[i].calculate_output(in_Kvectors=k_means_vectors)

        return self.layers[-1].calculate_output()

    # Backpropgate through the network
    def backpropagate(self, network_output, true_value):
        # Output layer delta
        self.layers[-1].delta_values = (network_output - true_value)

        # Hidden layer delta
        for i in reversed(range(1, self.num_layers - 1)):
            # No deltas for the bias values
            w_mod = self.layers[i].weights[0:-1, :]
            self.layers[i].delta_values = w_mod.dot(self.layers[i + 1].delta_values) * self.layers[i].derivatives

    def update_weights(self, learning_rate):
        for i in range(self.num_layers-1):
            weight_change = -learning_rate * np.outer(self.layers[i+1].delta_values, self.layers[i].outputs).T
            self.layers[i].weights += weight_change
