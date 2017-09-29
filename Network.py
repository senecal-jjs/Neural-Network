from Layer import layer
import random
from typing import Sequence

class Network(object):

    def _gen_weight_matrix(num_inputs, num_outputs, w_range):
        return numpy.array( [[0 if start == end else random.uniform(-0.5,0.5) for start in range(0,num_inputs)] for end in range(0,num_outputs)], float)


    def __init__(self, num_layers, neurons_per_layer, num_inputs, num_outputs=1):
        """Initialize a nueural net that is ready to train.
        Uses random a random value to initialize the weights
        """
        self.layers = []
        self.num_layers = num_layers

        # gen the layers:
        # no idea what value this is supposed to be:
        learning_rate = .5
        def gen_layer(num_inputs, num_outputs, last_layer):
            rweights = _get_weight_matrix(num_inputs, num_outputs, (-0.5,0.5))
            return layer(last_layer,num_outputs, rweights,learning_rate)

        #first layer:
        self.layers.append(gen_layer(num_inputs, neurons_per_layer, False))

        for i in range(1,num_layers-1):
            #assume that each layer fully connects to the next:
            self.layers.append(neurons_per_layer, neurons_per_layer, False)
        # get the output layer:
        self.layers.append(gen_layer(neurons_per_layer, num_outputs, True))
