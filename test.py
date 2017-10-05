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

    net = MLP.network([1, 35, 35, 1], "sigmoid")
    print("Training Network:")
    for i in range(x):
        data = gen_test_set(30,func)
        # net.train_batch(data, 0.0005)
        net.train_stochastic(data, 10, 3, 0.005)

    print("Testing Network:")
    set_size = 500
    # test how "good" it is:
    data = gen_test_set(set_size,func)
    sum_err = 0

    for i in data:
        output = net.calculate_outputs(i.inputs)
        # does root-mean squared error:
        err = output - i.solution
        # print("Error was: %f" % err)
        sum_err = sum_err + err**2

    print("RMSE: %f" % math.sqrt(sum_err/set_size))
