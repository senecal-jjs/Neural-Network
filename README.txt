The neural network GUI first presents the user with a drop down menu providing a choice of...
    1) Perceptron
    2) Radial Basis
    
If "Perceptron" is selected a second GUI will appear providing the user with the data entry fields...
    1) Number of inputs: The number of parameters, or attributes being provided to the network
    2) Number of outputs: The number of neurons in the output layer, always 1 for function approximation
    3) Number of examples: The number of training data examples
    4) Maximum iterations: The maximum training iterations to run if the network does not converge
    5) Hidden layers: The number of hidden layers to use in the network
    6) Number of Nodes: The number of nodes to use in the hidden layers. Specified as comma separated values
       For example: 3, 7 would create a network with 3 nodes in the 1st hidden layer and 7 nodes in the 2nd 
                    hidden layer
    7) Activation Function: Drop down menu that provides choice of sigmoid or hyperbolic tangent function
    8) Learning Rate: The learning rate used in weight updates during network training
    9) Momentum: If box is checked momentum will be incorporated in the weight updates
    10) Beta: The parameter used to set the influence of momentum in the weight updates. Typically between
              0.5 and 1. 
    11) Write Output: If box is checked the results of the experimental run will be written to a csv file. 
    
If "Radial Basis" is selected a second GUI will appear providing the user with the data entry fields...
    1) Number of inputs: The number of parameters, or attributes being provided to the network
    2) Number of outputs: The number of neurons in the output layer, always 1 for function approximation
    3) Number of examples: The number of training data examples
    4) Maximum iterations: The maximum training iterations to run if the network does not converge
    5) Number of gaussians: The number of basis functions to use in the network
    6) Learning Rate: The learning rate used in weight updates during network training
    7) Momentum: If box is checked momentum will be incorporated in the weight updates
    8) Beta: The parameter used to set the influence of momentum in the weight updates. Typically between
              0.5 and 1. 
    9) Write Output: If box is checked the results of the experimental run will be written to a csv file. 
    
To run the network: 
1) Enter all required parameters
2) Click approximate function