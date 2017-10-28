from numpy import *
from sigmoid import sigmoid

def predict(Theta, X, labels_test):
    # Takes as input a number of instances and the network learned variables
    # Returns a vector of the predicted labels for each one of the instances
    
    # Useful values
    m = X.shape[0]
    num_labels = Theta[-1].shape[0]
    num_layers = len(Theta) + 1

    # ================================ TODO ================================
    # You need to return the following variables correctly
    p = zeros((1,m))

    print(Theta)

    weights = []
    biases = []
    for x in range(0,num_layers-1):
        weights.append(Theta[x].transpose()[1:])
        biases.append(Theta[x].transpose()[0])

    index = 0
    for current_matrix in X:
        for i in range(0,num_layers-1):
            current_matrix = dot(current_matrix, weights[i])
            current_matrix = current_matrix + biases[i]

        p[0][index] = argmax(current_matrix)
        # print(current_matrix)
        # print(str(argmax(current_matrix)) + ' || ' + str(labels_test[index]))
        index = index + 1

    return p

