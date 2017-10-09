from tkinter import *
import buildGUI
import numpy as np
import random
import MLP

class buildNNMenu(Frame):
    def __init__(self, master = None):
        Frame.__init__(self, master)
        self.master = master
        self.init_gui()

    def init_gui(self):
        self.master.title('Neural Net Function Approximation')
        self.pack(fill=BOTH, expand=1)

        #Neural net selection
        menuLabel = Label(self, text="Select a neural net")
        menuLabel.grid(row=0, column=0)

        menuOptions = ["Feed-forward", "Radial Basis"]
        variable = StringVar(self.master)
        variable.set("               ")

        self.w = OptionMenu(self, variable, *menuOptions, command = self.selectNN)
        self.w.grid(row = 0, column = 1)

    def selectNN(self, value):

        root = Tk()
        app = buildGUI.buildGUI(value, root)
        root.mainloop()
        exit()

        


if __name__ == '__main__':
    root = Tk()
    app = buildNNMenu(root)
    root.mainloop()

    # Network initialization: ([No. Inputs, Neurons per Layer...], "activation function")
    '''net = MLP.network([2, 35, 35, 1], "sigmoid")


    # Test routine

    # Number of training examples
    x=100

    # Generate data and shuffle
    data = np.zeros((x, 2))
    data[:, 0] = np.linspace(1, 5, x)
    data[:, 1] = np.linspace(1, 5, x)
    np.random.shuffle(data)

    # x^2 function
    label = data[:, 0] * data[:, 1]

    print("Inputs")
    print(str(data[0, :]) + "\n")

    print("True Value")
    print(str(label[0]) + "\n")

    print("Inputs")
    print(str(data[1, :]) + "\n")

    print("True Value")
    print(str(label[1]) + "\n")

    # Run 300 training iterations over the data
    for k in range(0, 300):
        for i in range(0, x):
            output = net.calculate_outputs(data[i, :])
            net.backpropagate(output, label[i])
            net.update_weights(0.01)

    print("Network Outputs")
    print(net.calculate_outputs(data[0, :]))
    print(net.calculate_outputs(data[1, :]))'''






