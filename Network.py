from Layer import layer
import random
import itertools
from typing import Sequence, List

class Network(object):

    # kind of inefficent, but hey, why not
    def _gen_weight_matrix(self, num_inputs, num_outputs, w_range):
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
            rweights = self._gen_weight_matrix(num_inputs, num_outputs, (-0.5,0.5))
            return layer(last_layer,num_outputs, rweights,learning_rate)

        #first layer:
        self.layers.append(gen_layer(num_inputs, neurons_per_layer, False))

        for i in range(1,num_layers-1):
            #assume that each layer fully connects to the next:
            self.layers.append(neurons_per_layer, neurons_per_layer, False)
        # get the output layer:
        self.layers.append(gen_layer(neurons_per_layer, num_outputs, True))


    def _calc_output(self, vector :numpy.ndarray) -> (float, List[numpy.ndarray]):
        layer_outputs = []
        layer_outputs.append(self.layers[0].calculate_output(vector))
        for l in self.layers[1:]:
            layer_outputs.append(l.calculate_output(layer_outputs[-1:]))
        return layer_outputs[-1:], layer_outputs

    def calc_output(self, vector :numpy.ndarray) -> float:
        return self._calc_output(vector)[0]

    def _error_func(needed, actual):
        return needed - actual[0]

            # performs backpropagation, once per call:
    def _calc_weight_changes(self, data_point :trial_run, error_func=_error_func) -> List[numpy.ndarray]:
        """ Performs backpropagation and gets how the weights should change
        so the changes can be applied or averaged
        """
        out, layer_outputs = self._calc_output(data_point.inputs)
        # need to go backward through the list, so create the list ahead of time
        layer_deltas = [None for i in range(self.num_layers)]

        # calc the error of the output node, must be a numpy array:
        layer_deltas[self.num_layers-1] = numpy.array([error_func(data_point.solution, out)])

        #loop over list backwards and get the delta values:
        for i, u in reversed(list(enumerate(layer_outputs[:-1]))):
            layer_deltas[i] = u.backpropagation(layer_deltas[i+1],self.layers[i+1])

        #we now have all the info to get the change in weight:
        changew = []
        # don't know what to do for the first one:
        changew.append(elem.get_weight_change(data_point.inputs))
        for i, elem in enumerate(layers[:1]):
            changew.append(elem.get_weight_change(layer_outputs[i-1]))
        return changew

    def train_incremental(training_data : Sequence[trial_run]):
        for elem in training_data:
            self.train_batch([elem])

    def train_batch(self, training_data : Sequence[trial_run]):
        """Trains one batch of data
        """
        # place to keep changes in weights:
        running_total = [numpy.zeros(l.weights.shape) for l in layers]
        for data_point in training_data:
            c = self._calc_weight_changes(data_point)
            for i, change in zip(range(self.num_layers), c):
                running_total[i] = numpy.sum(running_total[i],change)

        # divide by the batch size:
        av_change = map(lambda x: numpy.divide(x, self.num_layers), running_total)
        # update the weights:
        for i in range(self.num_layers):
            self.layers[i].update_weights(av_change[i])
