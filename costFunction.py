from numpy import *
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient
from roll_params import roll_params
from unroll_params import unroll_params

def cross_entropy_loss(y, hx):
    cross_entropy = y*log(hx) + ((1 - y)*log(1 - hx))
    cross_entropy = cross_entropy*-1
    return cross_entropy

def compute_regularization_factor(weights, lambd, total_amount_of_samples):
    total_sum_squared_weights = 0
    for weight_matrix in weights:
        squared_weights = square(weight_matrix)
        total_sum_squared_weights = total_sum_squared_weights + sum(squared_weights)

    regularization_factor = (total_sum_squared_weights*lambd)/(2*total_amount_of_samples)

    return regularization_factor

def costFunction(nn_weights, layers, X, y, num_labels, lambd):
    # Computes the cost function of the neural network.
    # nn_weights: Neural network parameters (vector)
    # layers: a list with the number of units per layer.
    # X: a matrix where every row is a training example for a handwritten digit image
    # y: a vector with the labels of each instance
    # num_labels: the number of units in the output layer
    # lambd: regularization factor
    
    # Setup some useful variables
    m = X.shape[0]
    num_layers = len(layers)
    total_amount_of_samples = len(y)

    # Unroll Params
    Theta = roll_params(nn_weights, layers)
    
    # You need to return the following variables correctly 
    J = 0;
    
    # ================================ TODO ================================
    # The vector y passed into the function is a vector of labels
    # containing values from 1..K. You need to map this vector into a 
    # binary vector of 1's and 0's to be used with the neural network
    # cost function.
    yv = zeros((m, num_labels))

    for x in range(0, m):
        yv[x][y[x]] = 1

    # ================================ TODO ================================
    # In this point calculate the cost of the neural network (feedforward)

    weights = []
    biases = []
    for x in range(0,len(layers)-1):
        weights.append(Theta[x].transpose()[1:])
        biases.append(Theta[x].transpose()[0])

    current_matrix = X
    for i in range(0,len(layers)-1):
        current_matrix = dot(current_matrix, weights[i])
        current_matrix = current_matrix + biases[i]
        current_matrix = sigmoid(current_matrix)

    total_cross_entropy_sum = 0
    for i in range(0,total_amount_of_samples):
        for j in range(0,num_labels):         
            total_cross_entropy_sum = total_cross_entropy_sum + cross_entropy_loss(yv[i][j], current_matrix[i][j])


    final_cross_entropy_loss = total_cross_entropy_sum/total_amount_of_samples

    regularization_factor = compute_regularization_factor(weights, lambd, total_amount_of_samples)

    J = final_cross_entropy_loss + regularization_factor

    print('Current Loss - ' + str(J))

    return J

    




