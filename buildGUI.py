from tkinter import *
import trainingArray
import MLP
import numpy as np
import Kmeans
import RBF


class buildGUI(Frame):
    def __init__(self, nnType, master = None):
        Frame.__init__(self, master)
        self.master = master
        self.nnType = nnType
        self.init_gui()

    def init_gui(self):
        self.master.title(self.nnType + " function approximation")
        self.pack(fill=BOTH, expand=1)


        # Entry for number of inputs
        inputsLabel = Label(self, text="Number of inputs")
        inputsLabel.grid(row=0, column=0)

        self.inputs = Entry(self)
        self.inputs.grid(row=0, column=1)

        # Entry for number of outputs
        outputsLabel = Label(self, text="Number of outputs")
        outputsLabel.grid(row=1, column=0)

        self.outputs = Entry(self)
        self.outputs.grid(row=1, column=1)

        # Entry for number of training examples
        examplesLabel = Label(self, text="Number of examples")
        examplesLabel.grid(row=2, column=0)

        self.examples = Entry(self)
        self.examples.grid(row=2, column=1)

        # Entry for number of tests to run
        testsLabel = Label(self, text="Number of tests")
        testsLabel.grid(row=3, column=0)

        self.tests = Entry(self)
        self.tests.grid(row=3, column=1)

        # Update method
        updateLabel = Label(self, text="Update method")
        updateLabel.grid(row=8, column=0)
        options = ["incremental", "batch", "stochastic"]
        self.update_method = StringVar(self.master)
        self.update_method.set("            ")

        self.w = OptionMenu(self, self.update_method, *options)
        self.w.grid(row = 8, column = 1)

        # Check box for whether or not to create a csv output file
        wo = IntVar()
        self.w = Checkbutton(self, text="Write output", variable=wo)
        self.w.grid(row=10, column=0)
        self.write_output = wo.get()


        if self.nnType == "Perceptron":
            # Entry for number of iterations
            iterationsLabel = Label(self, text="Training iterations")
            iterationsLabel.grid(row=4, column=0)

            self.iterations = Entry(self)
            self.iterations.grid(row=4, column=1)

            # Number of hidden layers
            hiddenLabel = Label(self, text="Hidden Layers")
            hiddenLabel.grid(row=5, column=0)

            self.hiddenLayers = Entry(self)
            self.hiddenLayers.grid(row=5, column=1)

            # Number of nodes per layer
            nodesLabel = Label(self, text="Number of Nodes")
            nodesLabel.grid(row=6, column=0)

            self.nodes = Entry(self)
            self.nodes.grid(row=6, column=1)

            # Activation function selection menu
            menuLabel = Label(self, text="Activation Function")
            menuLabel.grid(row=7, column=0)

            menuOptions = ["sigmoid", "hyperbolic"]
            self.actFunc = StringVar(self.master)
            self.actFunc.set("              ")

            self.w = OptionMenu(self, self.actFunc, *menuOptions)
            self.w.grid(row = 7, column = 1)

            # Learning rate
            learningLabel = Label(self, text="Learning Rate")
            learningLabel.grid(row=9, column=0)

            self.learningRate = Entry(self)
            self.learningRate.grid(row=9, column=1)

        if self.nnType == "Radial Basis":
            # Entry for number of iterations
            iterationsLabel = Label(self, text="Training iterations")
            iterationsLabel.grid(row=4, column=0)

            self.iterations = Entry(self)
            self.iterations.grid(row=4, column=1)

            # Number of Gaussians to use
            gaussianLabel = Label(self, text="Number of Gaussians (k)")
            gaussianLabel.grid(row=5, column=0)

            self.gaussians = Entry(self)
            self.gaussians.grid(row=5, column=1)

            #Check box for whether or not to use K means clustering
            self.use_k_means = IntVar()
            self.c = Checkbutton(self, text="K-Means", variable=self.use_k_means)
            self.c.grid(row=6, column=0)

            #Learning rate
            learningLabel = Label(self, text="Learning Rate")
            learningLabel.grid(row=7, column=0)

            self.learningRate = Entry(self)
            self.learningRate.grid(row=7, column=1)


        #Button to start the neural net function approximation
        approximateButton = Button(self, text="Approximate Function", command=self.approx_function)
        approximateButton.grid(row=11, column=1)

    def approx_function(self):

        dataHandler = trainingArray.trainingArray(int(self.inputs.get()), int(self.examples.get()))
        self.data = dataHandler.createTrainingData()

        split = int((len(self.data) / 3) * 2)
        self.training_data = self.data[:split]
        self.testing_data = self.data[split:]

        if self.nnType == "Perceptron":
            self.run_mlp()

        if self.nnType == "Radial Basis":
            self.run_rbf()

        exit()

    def run_mlp(self):
        print("Starting MLP\n------------------------------------------------")
        # Print out what was just done:
        print("Number of inputs: %s" % self.inputs.get())
        print("Number of outputs: %s" % self.outputs.get())
        print("Number of examples: %s" % self.examples.get())
        print("Hidden Layers: %s" % self.hiddenLayers.get())
        print("Nodes per hidden layer: %s" % self.nodes.get())
        print("Activation function: %s" % self.actFunc.get())
        print("Update method: %s" % self.update_method.get())
        print("Learning rate: %s" % self.learningRate.get())
        print("Training iterations: %s\n" % self.iterations.get())

        # Set the number of nodes per layer as input for the MLP net
        net_layers = self.get_mlp_layers();
        net = MLP.network(net_layers, self.actFunc.get())
        net = self.train_mlp(net)
        self.test_network(net)


    def run_rbf(self):
        print("Starting RBF\n------------------------------------------------")
        # Print out what was just done:
        print("Number of inputs: %s" % self.inputs.get())
        print("Number of outputs: %s" % self.outputs.get())
        print("Number of examples: %s" % self.examples.get())
        print("Learning rate: %s" % self.learningRate.get())

        net_layers = self.get_rbf_layers()
        centroids = self.get_rbf_centroids()
        net = RBF.network(net_layers, "gaussian", centroids)

        for i in range(int(self.iterations.get())):
            net.train_incremental(self.training_data, float(self.learningRate.get()))

        self.test_network(net)

    def get_mlp_layers(self):
        ''' Return the array of number of nodes per layer for the MLP network'''
        net_layers = [int(self.inputs.get())]

        for lay in range(int(self.hiddenLayers.get())):
            net_layers.append(int(self.nodes.get())) 

        net_layers.append(int(self.outputs.get()))

        return net_layers

    def get_rbf_layers(self):
        ''' Return the array of number of nodes per layer in the RBF network '''
        net_layers = [int(self.inputs.get()), int(self.gaussians.get()), 
                      int(self.outputs.get())]

        return net_layers

    def get_rbf_centroids(self):
        ''' Given the method for selecting the k centroids, return an array
            of k centroids '''

        if self.use_k_means == 1:
            # centroids = Kmeans.kMeans(int(self.gaussians.get()), self.training_data, int(self.inputs.get()))
            pass
        else:
            centroids = []
            indices = []
            for i in range(len(self.training_data)):
                indices.append(i)

            sample_indices = np.random.choice(indices, int(self.gaussians.get()), replace = False)
            for ind in sample_indices:
                centroids.append(self.training_data[int(ind)].inputs)

        return centroids

    def train_mlp(self, mlp_net):
        ''' Given the network, iterations, and update method, train the net '''
        net = mlp_net
        learning = float(self.learningRate.get())

        for i in range(int(self.iterations.get())):

            if i % 100 == 0:
                print ("Beginning iteration " + str(i) + " of " + self.iterations.get() + "...")

            if self.update_method.get() == "incremental":
                net.train_incremental(self.training_data, learning)

            elif self.update_method.get() == "batch":
                net.train_batch(self.training_data, learning)

            elif self.update_method.get() == "stochastic":

                batch_size = int(np.sqrt(len(self.testing_data)))
                num_batches = int(int(self.iterations.get()) / batch_size)
                net.train_stochastic(self.training_data, batch_size, num_batches, learning)

        return net

    def test_network(self, net):
        ''' Given the trained net, calculate the output of the net
            Print the root mean square error to the console by default
            If write output is set, create a CSV with the test inputs,
            outputs, and other statistics '''

        input_vals = []
        output_vals = []
        true_vals = [test.solution for test in self.testing_data]

        for testInput in self.testing_data:
            data_in = testInput.inputs
            out_val = net.calculate_outputs(data_in)[0]
            output_vals.append(out_val)

        error = self.rmse(output_vals, true_vals)
        print ("RMSE: %f\n" % error)

        if self.write_output == 1:
            self.create_csv(inputs, output_vals);

    def rmse(self, predicted, true):
        ''' Given arrays of predicted and true values, calculate
            root mean square error '''

        return np.sqrt(((np.array(predicted) - np.array(true)) ** 2).mean())

    def create_csv(self, inputs, outputs, true):
        ''' Create a csv file with the test inputs, calculated outputs,
            true values and relevant statistics. '''

        print("Writing output...")