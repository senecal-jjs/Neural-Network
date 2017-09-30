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
    #root = Tk()
    #app = buildNNMenu(root)
    #root.mainloop()
    net = MLP.network([1, 10, 10, 1], "sigmoid")

    for k in range(0, 10):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        label = data * data
        index = random.randint(0, 9)

        for i in range(0, 100):
            output = net.calculate_outputs(data[7])
            net.backpropagate(output, label[7])
            net.update_weights(0.1)

    print(net.calculate_outputs(np.array([8])))
    #print(net.calculate_outputs(np.array([3])))






