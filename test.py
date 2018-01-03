#!/usr/bin/python3

import numpy as np
import random
import MLP
import RBF
import math
import trainingArray
import Kmeans
import sys
import numpy as np
from trainingArray import trial_run

def gen_test_setGGGGGG(size, func):
    data = []
    for i in range(size):
        val = np.random.randint(low=0, high=10)
        data.append(trial_run(inputs=val, solution=func(val)))
    return data;


def get_rbf_layers(inputs, gaussians, outputs):
    ''' Return the array of number of nodes per layer in the RBF network '''
    net_layers = [inputs, gaussians,
                  outputs]
    return net_layers

def train_rbf(rbf_net, training_data, learning_rate, num_iterations):
    for i in range(num_iterations):
        if i % 100 == 0:
            pass
            # print("Beginning iteration %d of %d..." % (i, num_iterations))
        np.random.shuffle(training_data)
        rbf_net.train_incremental(training_data, learning_rate)

def test_network(net, testing_data):
    ''' Given the trained net, calculate the output of the net
    # Print the root mean square error to the console by default
    If write output is set, create a CSV with the test inputs,
    outputs, and other statistics '''

    input_vals = []
    output_vals = []
    true_vals = [test.solution for test in testing_data]

    for testInput in testing_data:
        data_in = testInput.inputs
        out_val = net.calculate_outputs(data_in)[0]
        output_vals.append(out_val)
        input_vals.append(data_in)

    error = rmse(output_vals, true_vals)
    # print ("RMSE: %f\n" % error)

    write = False
    if write:
        pass
        # self.create_csv(input_vals, output_vals, true_vals);
    return error

def rmse(predicted, true):
    ''' Given arrays of predicted and true values, calculate
    root mean square error '''

    return np.sqrt(((np.array(predicted) - np.array(true)) ** 2).mean())


if __name__ == '__main__':
    # number of training sets:
    num_inputs = int(sys.argv[2])
    num_outputs = 1
    num_examples = int(sys.argv[1])
    learning_rate = 0.005
    num_iterations = 500
    # gauss = int(sys.argv[2])
    num_trials = 100

    total = 0

    k_vals = [10, 50, 100, 500, 1000]

    for k in k_vals:
        print("For k-val: %d" % k)
        gauss = k
        total = 0
        for i in range(num_trials):
            # print("Building training array")
            dataHandler = trainingArray.trainingArray(num_inputs, num_examples)
            data = dataHandler.createTrainingData()

            split = int((len(data) / 3) * 2)
            training_data = data[:split]
            testing_data = data[split:]

            training_inputs = [example.inputs for example in training_data]
            # print("Computing kmeans")
            centroids = Kmeans.kMeans(gauss, training_inputs, num_inputs).calculateKMeans()

            net_layers = get_rbf_layers(num_inputs, gauss, num_outputs)

            # print("Centroids computed!\n")
            # net = MLP.network([1, 35, 35, 1], "sigmoid")
            net = RBF.network(net_layers, "gaussian", centroids)

            # print("Training Network:")

            train_rbf(net,training_data, learning_rate, num_iterations)

            result = test_network(net, testing_data)
            total += result
            print(result)
            # print("\n")

        print("Average: %f" % (total / num_trials))
