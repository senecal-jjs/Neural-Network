import numpy as np
import Neuron


class layer:
    def __init__(self, weight_size, activation_function, input_layer = False, output_layer = False):
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.weight_size = weight_size  # (neurons in prev layer, neurons in next layer)

        # Holds outputs and inputs for the layer
        self.outputs = np.zeros(weight_size[0])
        self.inputs = None

        # Create neuron for output calculations
        self.neuron = Neuron.neuron(activation_function, sigma=None)

        # Create matrices to hold weights, deltas, and derivatives
        self.weights = None
        self.delta_values = None
        self.derivatives = None

        if not input_layer:
            self.inputs = np.zeros(weight_size[0])
            self.delta_values = np.zeros(weight_size[0])

        if not output_layer:
            self.weights = np.random.uniform(-0.2, 0.2, size=weight_size)

        if not output_layer and not input_layer:
            self.derivatives=np.zeros(weight_size[0])


    # Calculate output for layer's neurons
    def calculate_output(self):
        if self.input_layer:
            return self.outputs.dot(self.weights)

        self.outputs = self.neuron.calculate_output(i_inputs=self.inputs, in_Kvectors=None)
        if self.output_layer:
            return self.outputs
        else:
            # For hidden layers add bias values, and calculate derivatives
            self.outputs = np.append(self.outputs, 1) # add 1 for bias activation
            self.derivatives = self.neuron.calculate_output(i_inputs=self.inputs, i_want_derivative=True)

            return self.outputs.dot(self.weights)

    def set_delta(self, in_delta):
        self.delta_values = in_delta