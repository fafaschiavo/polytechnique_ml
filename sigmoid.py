from numpy import *

def sigmoid(z):

    # SIGMOID returns sigmoid function evaluated at z

    # ============================= TODO ================================
    # Instructions: Compute sigmoid function evaluated at each value of z.
    g = 1/(1+exp(-z))

    return g
