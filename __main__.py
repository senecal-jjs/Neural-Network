from tkinter import *
import buildGUI
import numpy as np
import random
import MLP
import RBF
import Kmeans
# import matplotlib.pyplot as plt

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

        menuOptions = ["Perceptron", "Radial Basis"]
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
    # test_input = []
    # layer_structure = np.array([2, 60, 1])
    # num_examples = 500
    # for i in range(num_examples):
    #     sample = []
    #     for j in range(layer_structure[0]):
    #         sample.append(np.random.uniform(-1, 1))
    #     test_input.append(sample)
    # parameters = Kmeans.kMeans(layer_structure[1], test_input, layer_structure[0])
    # testK = parameters.calculateKMeans()
    # print("K means calculated!")
    #
    # x = [seq[0] for seq in test_input]
    # y = [seq[1] for seq in test_input]
    # xk = [seq[0] for seq in testK]
    # yk = [seq[1] for seq in testK]
    # plt.scatter(x, y, color='k')
    # plt.scatter(xk, yk, color='r')
    #
    # net = RBF.network(layer_structure, "gaussian", testK)
    #
    # label = np.zeros(num_examples)
    # for i in range(len(label)):
    #     # Rosenbrock function
    #     label[i] = (1 - test_input[i][0]) ** 2 + 100 * (test_input[i][1] - test_input[i][0] ** 2) ** 2
    #
    # for j in range(150):
    #     if j % 50 == 0: print("Iteration completed!")
    #     for i in range(len(test_input)):
    #         output = net.calculate_outputs(test_input[i])
    #         net.backpropagate(output, label[i])
    #         net.update_weights(0.01)
    #
    # print("True Value")
    # print(label[2])
    # print(net.calculate_outputs(test_input[2]))
    # print(label[6])
    # print(net.calculate_outputs(test_input[6]))
    # print(label[70])
    # print(net.calculate_outputs(test_input[70]))
    # plt.show()
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






