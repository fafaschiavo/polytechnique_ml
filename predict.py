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

    weights = []
    biases = []
    for x in range(0,num_layers-1):
        weights.append(Theta[x].transpose()[1:])
        biases.append(Theta[x].transpose()[0])

    index = 0
    for current_matrix in X:
        # print('---------------- X')
        # print(current_matrix.shape)
        # print('----------------')
        # for i in range(0, len(weights)-1):
        #     print(weights[i].shape)
        #     print(biases[i].shape)
        # print('----------------')
        feed_forward_matrix = current_matrix
        for i in range(0,num_layers-1):
            feed_forward_matrix = dot(feed_forward_matrix, weights[i])
            feed_forward_matrix = feed_forward_matrix + biases[i]
            feed_forward_matrix = sigmoid(feed_forward_matrix)

        p[0][index] = argmax(feed_forward_matrix)
        # print(current_matrix)
        # print(str(argmax(current_matrix)) + ' || ' + str(labels_test[index]))
        index = index + 1

    return p

