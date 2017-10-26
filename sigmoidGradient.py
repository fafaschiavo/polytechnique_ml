from numpy import *
from sigmoid import sigmoid

def sigmoidGradient(z):

    # SIGMOIDGRADIENT returns the gradient of the sigmoid function evaluated at z


    # g = zeros(z.shape)
    # =========================== TODO ==================================
    # Instructions: Compute the gradient of the sigmoid function evaluated at
    #               each value of z.
    g = exp(-z)/((1+exp(-z))**2)

    return g




