from numpy import *

def randInitializeWeights(layers):

    num_of_layers = len(layers)
    epsilon = 0.12
        
    Theta = []

    for i in range(num_of_layers-1):
        # ====================== TODO ======================
        # Instructions: Initialize W randomly so that we break the symmetry while
        #               training the neural network.
        #
        W = random.rand(layers[i+1], layers[i] + 1)
        W = W/10
        Theta.append(W)
                
    return Theta
            
