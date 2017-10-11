import numpy as np
import Layer
import math
from typing import Sequence
from trainingArray import trial_run

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
            self.layers[i+1].inputs = self.layers[i].calculate_output()

        return self.layers[-1].calculate_output()

    def backpropagate(self, network_output, true_value):
        self.layers[-1].delta_values = np.transpose(network_output - true_value)

        for i in reversed(range(1, self.num_layers - 1)):
            # No deltas for the bias values
            w_mod = self.layers[i].weights[0:-1,:]
            self.layers[i].delta_values = w_mod.dot(self.layers[i + 1].delta_values) * self.layers[i].derivatives

    def calc_update_weights(self, learning_rate):
        weight_changes = []
        for i in range(self.num_layers-1):
            weight_change = -learning_rate * np.outer(self.layers[i+1].delta_values, self.layers[i].outputs).T
            weight_changes.append(weight_change);
        return weight_changes


    def update_weights(self, weight_changes):
        for i in range(self.num_layers-1):
            self.layers[i].weights += weight_changes[i]


    def train_until_convergence(self, training_data : Sequence[trial_run], testing_data : Sequence[trial_run],
                                train_func, threshold, learning_rate,
                                stop_iter=10000):
        """Uses the root mean squared error calculate the error of the function. When the error
        falls below threshold, the training stops. The training also stops if there have been
        stop_iter iterations.
        """
        iter_num = 0
        # some value greater than the threshold (I hope:):
        error = 10000
        while((iter_num == 0 or error > threshold) and iter_num <= stop_iter):
            train_func(training_data, learning_rate)
            sum_err = 0
            for i in testing_data:
                output = self.calculate_outputs(i.inputs)
                # does root-mean squared error:
                e = output - i.solution
                sum_err = sum_err + e**2
            error = math.sqrt(sum_err/len(testing_data))
            iter_num = iter_num + 1
        # Tell the caller about why we stopped:
        if error <= threshold:
            return True
        else:
            return False


    def train_batch(self, training_data : Sequence[trial_run], learning_rate):
        """Trains one batch of data
        """
        # place to keep changes in weights:
        running_total = [np.zeros(l.weights.shape) for l in self.layers[:-1]]
        for data_point in training_data:
            output = self.calculate_outputs(data_point.inputs)
            # internally calculate the delta values:
            self.backpropagate(output, data_point.solution)
            # get the change in weight from those delta values:
            change = self.calc_update_weights(learning_rate)
            # sum the changes:
            for i in range(len(change)):
                running_total[i] = running_total[i] + change[i]

        assert len(change) == self.num_layers-1
        # divide by the batch size:
        av_change = list(map(lambda x: np.divide(x, self.num_layers), running_total))
        # update the weights:
        self.update_weights(av_change)

    def train_incremental(self, training_data : Sequence[trial_run], learning_rate):
        """Applies all the elements one at a time to the data set, and updates
        the weights after every data point
        """
        for d in training_data:
            self.train_batch([d],learning_rate)

    def train_stochastic(self, training_data : Sequence[trial_run], batch_size, num_batches, learning_rate):
        """Trains a series of random batches from training_data of size batch_size.
        Repeats num_batches times.
        """
        for i in range(num_batches):
            t_set_indices = np.random.choice(range(len(training_data)),batch_size,replace=False)
            t_set = [training_data[i] for i in t_set_indices]
            self.train_batch(t_set, learning_rate)
