#!/usr/bin/python3

import numpy as np
import random
import MLP
import math
from trainingArray import trial_run

def gen_test_set(size, func):
    data = []
    for i in range(size):
        val = np.random.randint(low=0, high=10)
        data.append(trial_run(inputs=val, solution=func(val)))
    return data;


if __name__ == '__main__':
    # number of training sets:
    x = 3000
    func = lambda inp: inp**2

    test_size = 1000
    train_size = 500

    net = MLP.network([1, 35, 35, 1], "sigmoid")


    test_set = gen_test_set(test_size,func)
    training_set = gen_test_set(train_size,func)

    def stochastic_auto(batch_size, num_batches):
        def run(training_data, learning_rate):
            net.train_stochastic(training_data, batch_size, num_batches, learning_rate)
        return run

    print("Training Network:")
    result = net.train_until_convergence(training_set, test_set, stochastic_auto(100,4), .3, .001, 5000)
    if result:
        print("Convergence during training occured")
    else:
        print("No convergence while training")

    print("Testing Network:")
    # test how "good" it is:
    sum_err = 0

    for i in test_set:
        output = net.calculate_outputs(i.inputs)
        # does root-mean squared error:
        err = output - i.solution
        # print("Error was: %f" % err)
        sum_err = sum_err + err**2

    print("RMSE: %f" % math.sqrt(sum_err/test_size))
