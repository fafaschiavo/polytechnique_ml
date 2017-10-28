from numpy import *
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient
from roll_params import roll_params
from unroll_params import unroll_params

def compute_regularization_factor(weights, lambd, total_amount_of_samples):
    total_sum_squared_weights = 0
    for weight_matrix in weights:
        squared_weights = square(weight_matrix)
        total_sum_squared_weights = total_sum_squared_weights + sum(squared_weights)

    regularization_factor = (total_sum_squared_weights*lambd)/(2*total_amount_of_samples)

    return regularization_factor

def backwards(nn_weights, layers, X, y, num_labels, lambd):
    # Computes the gradient fo the neural network.
    # nn_weights: Neural network parameters (vector)
    # layers: a list with the number of units per layer.
    # X: a matrix where every row is a training example for a handwritten digit image
    # y: a vector with the labels of each instance
    # num_labels: the number of units in the output layer
    # lambd: regularization factor
    
    # Setup some useful variables
    m = X.shape[0]
    num_layers = len(layers)
    accumulated_gradient = []
    accumulated_bias_gradient = []
    first_iteration = True

    # Roll Params
    # The parameters for the neural network are "unrolled" into the vector
    # nn_params and need to be converted back into the weight matrices.
    Theta = roll_params(nn_weights, layers)
  
    # You need to return the following variables correctly 
    Theta_grad = [zeros(w.shape) for w in Theta]

    # ================================ TODO ================================
    # The vector y passed into the function is a vector of labels
    # containing values from 1..K. You need to map this vector into a 
    # binary vector of 1's and 0's to be used with the neural network
    # cost function.
    yv = zeros((m, num_labels))

    for x in range(0, m):
        yv[x][y[x]] = 1

    # ================================ TODO ================================
    # In this point implement the backpropagaition algorithm 

    # Unroll Params
    Theta_grad = unroll_params(Theta_grad)

    weights = []
    biases = []
    for x in range(0,len(layers)-1):
        weights.append(Theta[x].transpose()[1:])
        biases.append(Theta[x].transpose()[0])

    for current_sample in range(0,len(X)):
        current_matrix = X[current_sample]
        current_y = yv[current_sample]
        a_arrays = [X[current_sample]]
        z_arrays = []
        for i in range(0,len(layers)-1):
            current_matrix = dot(current_matrix, weights[i])
            current_matrix = current_matrix + biases[i]
            z_arrays.append(current_matrix)
            current_matrix = sigmoid(current_matrix)
            a_arrays.append(current_matrix)

        previous_sigma = a_arrays[-1]-yv[current_sample]
        sigma_array = [previous_sigma]

        for x in reversed(range(0,len(layers)-2)):
            new_sigma = weights[x+1]
            new_sigma = dot(new_sigma, previous_sigma)
            new_sigma = multiply(new_sigma, sigmoidGradient(z_arrays[x]))
            sigma_array = [new_sigma] + sigma_array
            previous_sigma = new_sigma

        for x in range(0,len(a_arrays)-1):
            current_sigma = expand_dims(sigma_array[x], 0).transpose()
            current_a = expand_dims(a_arrays[x], 0)

            current_gradient = dot(current_sigma, current_a)
            if first_iteration:
                accumulated_gradient.append(current_gradient/m)
                accumulated_bias_gradient.append(sigma_array[x]/m)
            else:
                accumulated_gradient[x] = accumulated_gradient[x] + (current_gradient/m)
                accumulated_bias_gradient[x] = accumulated_bias_gradient[x] + (sigma_array[x]/m)
        
        first_iteration = False
        
    accumulated_gradient_transposed = []
    for x in range(0,len(a_arrays)-1):
        current_layer_accumulated_gradient = concatenate((expand_dims(accumulated_bias_gradient[x],0).transpose(), accumulated_gradient[x]), axis=1)
        current_layer_regularization_factor = concatenate((expand_dims(zeros(biases[x].shape),0).transpose(), weights[x].transpose()), axis=1)
        current_layer_regularization_factor = current_layer_regularization_factor * (lambd/m)
        current_layer_accumulated_gradient = current_layer_accumulated_gradient + current_layer_regularization_factor
        accumulated_gradient_transposed.append(current_layer_accumulated_gradient)

    Theta_grad = unroll_params(accumulated_gradient_transposed)

    return Theta_grad

    
