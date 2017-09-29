import numpy as np
import Neuron

class layer:
    def __init__(self, last_layer, num_neurons, weight_matrix, learning_rate):
        self.last_layer = last_layer
        self.num_neurons = num_neurons
        self.weights = weight_matrix
        self.outputs = np.zeros([num_neurons]) # 1 added for bias activation
        self.delta_values = np.zeros([num_neurons])
        self.learning_rate = learning_rate
        self.neuron = Neuron.neuron()  # Used to calculate outputs

    # Calculate output for layer's neurons
    def calculate_output(self, prev_layer_outputs):
        for i in range(self.num_neurons):
            self.outputs[i] = self.neuron.calculate_output(prev_layer_outputs, self.weights[i, :], self.last_layer)

    # Execute back_propagation step for the layer
    def back_propagation(self, downstream_deltas, downstream_weights):
        # Update delta values
        for i in range(self.num_neurons):
            if len(downstream_deltas) == 1:
                self.delta_values[i] = self.outputs[i] * (1 - self.outputs[i]) * np.dot(downstream_deltas,downstream_weights[:,i])
            else:
                self.delta_values[i] = self.outputs[i]*(1-self.outputs[i])*np.dot(downstream_deltas, downstream_weights[:,i])

        return self.delta_values

    def get_weight_change(self, upstream_layer_outputs):
        # create a numpy matrix to hold the changes:
        results = np.zeros(shape=(self.num_neurons, self.len(upstream_layer_outputs)))
        # Update weights
        for row in range(self.num_neurons):
            for col in range(len(upstream_layer_outputs)):
                results[row][col] += self.learning_rate*self.delta_values[row]*upstream_layer_outputs[col]
        return results

    def update_weights(self,change_in_weights):
        # Update weights
        self.weights = np.add(self.weights, change_in_weights)

    def get_weight(self):
        return self.weights

    def get_outputs(self):
        return self.outputs

    def get_delta(self):
        return self.delta_values

    def get_last_layer(self):
        return self.last_layer

    def set_delta(self, in_delta):
        self.delta_values = in_delta