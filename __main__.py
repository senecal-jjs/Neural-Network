from tkinter import *
import urllib3
import buildGUI


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
