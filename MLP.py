import numpy as np
import Layer
from types import Sequence

class network:
    # include number of inputs as the first value in neurons_per_layer
    def __init__(self, neurons_per_layer, activation_function):
        self.layers = []
        self.num_layers = len(neurons_per_layer)

        for i in range(self.num_layers-1):
            # Activate Input layer, the +1 in neurons_per_layer[i]+1 is to hold a bias value
            if i == 0:
                self.layers.append(Layer.layer([neurons_per_layer[i] + 1, neurons_per_layer[i + 1]], "linear", input_layer=True))

            # Activate Hidden Layers
            else:
                self.layers.append(Layer.layer([neurons_per_layer[i] + 1, neurons_per_layer[i + 1]], activation_function))

        # Activate output layer
        self.layers.append(Layer.layer([neurons_per_layer[-1], None], "linear", output_layer=True))

    # inputs is a one dimensional array containing the inputs to the function
    def calculate_outputs(self, inputs):
        self.layers[0].outputs = np.append(inputs, 1)  # the 1 is added as a bias value

        for i in range(self.num_layers - 1):
            #print("input" + str(i))
            #print(self.layers[i].inputs)
            self.layers[i+1].inputs = self.layers[i].calculate_output()

        return self.layers[-1].calculate_output()

    def backpropagate(self, network_output, true_value):
        self.layers[-1].delta_values = np.transpose(network_output - true_value)

        for i in reversed(range(1, self.num_layers - 1)):
            # No deltas for the bias values
            w_mod = self.layers[i].weights[0:-1,:]
            self.layers[i].delta_values = w_mod.dot(self.layers[i + 1].delta_values) * \
                                                    self.layers[i].derivatives

    def calc_update_weights(self, learning_rate):
        weight_changes = []
        for i in range(self.num_layers-1):
            weight_change = -learning_rate * np.outer(self.layers[i+1].delta_values, self.layers[i].outputs).T
            #print()
            #print("weight change")
            #print(weight_change)
            weight_changes.append(weight_change);
        return weight_changes


    def update_weights(self, weight_changes):
        for i in range(self.num_layers-1):
            self.layers[i].weights += weight_changes[i]


    def train_batch(self, training_data : Sequence[trial_run], learning_rate):
        """Trains one batch of data
        """
        # place to keep changes in weights:
        running_total = [numpy.zeros(l.weights.shape) for l in layers]
        for data_point in training_data:
            output = self.calculate_outputs(data_point.inputs)
            # internally calculate the delta values:
            self.backpropagate(output, data_point.solution)
            # get the change in weight from those delta values:
            change = self.calc_update_weights(learning_rate)
            # sum the changes:
            for i in range(len(change)):
                running_total[i] = numpy.sum(running_total[i], change[i]);

        assert len(change) == self.num_layers
        # divide by the batch size:
        av_change = map(lambda x: numpy.divide(x, self.num_layers), running_total)
        # update the weights:
        for i in range(self.num_layers):
            self.layers[i].update_weights(av_change[i])
