# Single Layer Perceptron implemented from scratch were you can see the 
# differences between perceptron learning, delta learning, batch learning 
# and sequential learning. Two classes from a normal distribution was generated
# to show the algorithm performance.

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

def generate_data(n_training_points=100, means_classA=[-0.5,-0.5], 
                  means_classB=[1.0, 3.0], sigma_classA=0.5, sigma_classB=0.5):
    """
    Generate data for binary classification from a normal distribution. 
    Class A has as target -1 and class B has target 1.

    Parameters
    ----------
    n_training_points : INT, default=100
    means_classA : NUMPY ARRAY, default=[-0.5,-0.5]
    means_classB : NUMPY ARRAY, default=[1.0, 3.0]
    sigma_classA : FLOAT, default=0.5
    sigma_classB : FLOAT, default=0.5

    Returns
    -------
    patterns_shuffled : NUMPY ARRAY (#dimensions,#training_samples)
    targets_shuffled : NUMPY ARRAY (#trainig_samples,)

    """
    
    dimensions_input_data = len(means_classA)
    patterns_classA = np.zeros((dimensions_input_data,n_training_points))
    patterns_classB = np.zeros((dimensions_input_data,n_training_points))
    targets_classA = -np.ones(n_training_points)
    targets_classB = np.ones(n_training_points)

    for i_dimension in range(dimensions_input_data):
        patterns_classA[i_dimension] = np.random.randn(1, n_training_points) * sigma_classA + means_classA[i_dimension]

    for i_dimension in range(dimensions_input_data):
        patterns_classB[i_dimension] = np.random.randn(1, n_training_points) * sigma_classB + means_classB[i_dimension]

    patterns = np.concatenate((patterns_classA, patterns_classB), axis=1)

    targets = np.concatenate((targets_classA, targets_classB), axis=0).reshape(1,-1)

    patterns_targets = np.concatenate((patterns, targets), axis=0)

    rand_idx_cols = np.random.permutation(patterns_targets.shape[1])

    patterns_targets_shuffled = patterns_targets[:,rand_idx_cols]

    patterns_shuffled = patterns_targets_shuffled[:-1]
    targets_shuffled = patterns_targets_shuffled[-1]
    
    return patterns_shuffled, targets_shuffled






def train_batch(patterns, targets, learning_rate=0.001, n_epochs=100, 
                type_of_learning = "delta", add_bias=True, not_optimal_W_init=False):
    """
    
    Parameters
    ----------
    patterns : NUMPY ARRAY (#dimensions,#training_samples)
    targets : NUMPY ARRAY (#training_samples,)
    learning_rate : FLOAT, default=0.001.
    n_epochs : INT, default=100.
    type_of_learning : STRING, default="delta".
    add_bias : BOOL, default=True.
    not_optimal_W_init: BOOL, default=False.

    Returns
    -------
    W : NUMPY ARRAY (1,#dimensions)

    """
    if add_bias:
        bias_row = np.ones(patterns.shape[1]).reshape(1,-1)
        patterns = np.concatenate((patterns, bias_row), axis=0)
        
    n_dimensions_training = patterns.shape[0]
    
    W = init_weights(n_dimensions_training,not_optimal_W_init)
    
    errors = []
    for epoch in range(n_epochs):        
        if type_of_learning == "delta":
            W = W + delta_learning(W, patterns, targets, learning_rate)
        elif type_of_learning == "perceptron":
            W = W + perceptron_learning(W, patterns, targets, learning_rate)

        errors.append(get_classification_acc(W, patterns, targets))
        
    plot_convergence(errors, "batch", type_of_learning)

    return W


def train_sequential(patterns, targets, learning_rate=0.001, n_epochs=100, 
                     type_of_learning = "delta", add_bias=True, not_optimal_W_init=False):
    """
    
    Parameters
    ----------
    patterns : NUMPY ARRAY (#dimensions,#training_samples)
    targets : NUMPY ARRAY (#training_samples,)
    learning_rate : FLOAT, default=0.001.
    n_epochs : INT, default=100.
    type_of_learning : STRING, default="delta".
    add_bias : BOOL, default=True.
    not_optimal_W_init: BOOL, default=False.
    
    Returns
    -------
    W : NUMPY ARRAY (1,#dimensions)

    """
    if add_bias:
        bias_row = np.ones(patterns.shape[1]).reshape(1,-1)
        patterns = np.concatenate((patterns, bias_row), axis=0)

    n_training_samples = patterns.shape[1]
    n_dimensions_training = patterns.shape[0]
    
    W = init_weights(n_dimensions_training, not_optimal_W_init)
    
    classification_acc_per_epoch = []
    for epoch in range(n_epochs):
        for datapoint_index in range(n_training_samples):
            this_pattern = patterns[:, datapoint_index].reshape(-1,1)
            this_target = targets[datapoint_index].reshape(-1,)
            if type_of_learning == "delta":
                W = W + delta_learning(W, this_pattern, this_target, learning_rate)
            elif type_of_learning == "perceptron":
                W = W + perceptron_learning(W, this_pattern, this_target, learning_rate)

        classification_acc_per_epoch.append(get_classification_acc(W, patterns, targets))

    plot_convergence(classification_acc_per_epoch, "sequential", type_of_learning)
    
    return W


def plot_convergence(classification_acc_per_epoch, batch_or_sequential_learning, delta_or_perceptron_learning):
    """
    Plot classification accuracy in each epoch.

    Parameters
    ----------
    classification_acc_per_epoch : LIST
    batch_or_sequential_learning : STRING
    delta_or_perceptron_learning : STRING

    Returns
    -------
    None.

    """
    plt.figure()
    plt.plot(classification_acc_per_epoch)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    txt_title = "Convergence " + batch_or_sequential_learning + " " + delta_or_perceptron_learning + " learning"
    plt.title(txt_title)
    plt.show()


def get_classification_acc(W, patterns, targets):
    """
    
    Parameters
    ----------
    W : NUMPY ARRAY ()
        DESCRIPTION.
    patterns : TYPE
        DESCRIPTION.
    targets : TYPE
        DESCRIPTION.

    Returns
    -------
    classification_acc : NUMPY ARRAY (#training_points,)
        Classification accuracy of each training sample.

    """
    prediction_before_step = (W @ patterns)[0]
    prediction = step_function(prediction_before_step)
    prediction_target_difference = prediction - step_function(targets)
    n_misclassifications = np.sum(np.abs(prediction_target_difference))
    missclassified_ratio = n_misclassifications/len(targets)
    classification_acc = 1 - missclassified_ratio
    return classification_acc


def step_function(Y):
    """
    Parameters
    ----------
    Y : NUMPY ARRAY (#training_samples,)

    Returns
    -------
    Y_after_step : NUMPY ARRAY (#training_samples,)

    """
    Y_after_step = np.zeros(len(Y))
    Y_after_step[Y>0] = 1
    return Y_after_step


def init_weights(n_dimensions_training, not_optimal_W_init, mu=0, sigma=0.1):
    """
    The weights of the algorithm are initialized optimally from a normal 
    distribution. If we prefer not optimal initialization the weights would be 
    a vector of ones. 

    Parameters
    ----------
    n_dimensions_training : INT
    not_optimal_W_init: BOOL 
    mu : INT, optional
    sigma : INT, optional

    Returns
    -------
    NUMPY ARRAY with shape (1, input_dimension)
    """
    if not_optimal_W_init:
        W = np.ones((1, n_dimensions_training))
    else:
        W = np.random.normal(mu, sigma, n_dimensions_training)
    
    return W.reshape(1,-1)


def delta_learning(W, patterns, targets, learning_rate):
    delta_W = -learning_rate * ((W @ patterns) - targets) @ np.transpose(patterns)
    return delta_W


def perceptron_learning(W, patterns, targets, learning_rate):   
    prediction_before_step =  (W @ patterns).reshape(-1,)
    prediction = step_function(prediction_before_step) 
    
    delta_W = -learning_rate * (prediction - step_function(targets)) @ np.transpose(patterns)
    
    return delta_W



def plot_result_training(patterns, targets, W_trained):

    plt.scatter(patterns[0],patterns[1], marker='x', c=targets)
    
    m = -(W_trained[0][0]/W_trained[0][1])
    q = - (W_trained[0][2]/W_trained[0][1])

    Z1 = np.linspace(-5, 5, 200)
    Z2 = m*Z1 + q
    plt.plot(Z1, Z2, "--k")       
    plt.show()
    

if __name__ == "__main__":
    print("Generating data...")
    patterns, targets = generate_data()
    
    print("Delta learning in batch ...")
    W_trained = train_batch(patterns, targets, type_of_learning="delta")
    plot_result_training(patterns, targets, W_trained)
    
    print("Perceptron learning in batch...")
    W_trained = train_batch(patterns, targets, type_of_learning="perceptron")
    plot_result_training(patterns, targets, W_trained)
    
    print("Online delta learning...")
    W_trained = train_sequential(patterns, targets, type_of_learning="delta")
    plot_result_training(patterns, targets, W_trained)
    
    print("Online perceptron learning...")
    W_trained = train_sequential(patterns, targets, type_of_learning="perceptron")
    plot_result_training(patterns, targets, W_trained)
    
    print("Delta learning in batch without an optimal initialization of weights (array of ones)...")
    W_trained = train_batch(patterns, targets, type_of_learning="delta", not_optimal_W_init=True)
