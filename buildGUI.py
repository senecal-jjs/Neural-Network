from tkinter import *
import urllib3
import trainingArray


class buildGUI(Frame):
    def __init__(self, nnType, master = None):
        Frame.__init__(self, master)
        self.master = master
        self.nnType = nnType
        self.init_gui()

    def init_gui(self):
        self.master.title(self.nnType + " function approximation")
        self.pack(fill=BOTH, expand=1)


        #Entry for number of inputs
        inputsLabel = Label(self, text="Number of inputs")
        inputsLabel.grid(row=0, column=0)

        self.inputs = Entry(self)
        self.inputs.grid(row=0, column=1)

        #Entry for number of outputs
        outputsLabel = Label(self, text="Number of outputs")
        outputsLabel.grid(row=1, column=0)

        self.outputs = Entry(self)
        self.outputs.grid(row=1, column=1)

        #Entry for number of training examples
        examplesLabel = Label(self, text="Number of examples")
        examplesLabel.grid(row=2, column=0)

        self.examples = Entry(self)
        self.examples.grid(row=2, column=1)

        if self.nnType == "Feed-forward":
            #Entry for number of iterations
            iterationsLabel = Label(self, text="Training iterations")
            iterationsLabel.grid(row=3, column=0)

            self.iterations = Entry(self)
            self.iterations.grid(row=3, column=1)

            #Number of hidden layers
            hiddenLabel = Label(self, text="Hidden Layers")
            hiddenLabel.grid(row=4, column=0)

            self.hiddenLayers = Entry(self)
            self.hiddenLayers.grid(row=4, column=1)

            #Number of nodes per layer
            nodesLabel = Label(self, text="Number of Nodes")
            nodesLabel.grid(row=5, column=0)

            self.nodes = Entry(self)
            self.nodes.grid(row=5, column=1)

            #Activation function selection menu
            menuLabel = Label(self, text="Activation Function")
            menuLabel.grid(row=6, column=0)

            menuOptions = ["Sigmoid", "Hyperbolic Tangent"]
            variable = StringVar(self.master)
            variable.set("                     ")

            self.w = OptionMenu(self, variable, *menuOptions)
            self.w.grid(row = 6, column = 1)

        if self.nnType == "Radial Basis":
            #Number of Gaussians to use
            gaussianLabel = Label(self, text="Number of Gaussians (k)")
            gaussianLabel.grid(row=3, column=0)

            self.gaussians = Entry(self)
            self.gaussians.grid(row=3, column=1)

            #Enter the value of sigma for Gaussians
            sigmaLabel = Label(self, text="Sigma")
            sigmaLabel.grid(row=4, column=0)

            self.sigma = Entry(self)
            self.sigma.grid(row=4, column=1)

            #Check box for whether or not to use K means clustering
            var = IntVar()
            self.c = Checkbutton(self, text="K-Means", variable=var)
            self.c.grid(row=5, column=0)


        #Button to start the neural net function approximation
        approximateButton = Button(self, text="Approximate Function", command=self.approxFunction)
        approximateButton.grid(row=7, column=1)

    def approxFunction(self):

        dataHandler = trainingArray.trainingArray(int(self.inputs.get()), int(self.examples.get()))
        trainingData = dataHandler.createTrainingData()

        for row in trainingData:
            print (row)

        exit()