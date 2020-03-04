# Perceptron implementation

In <b>SLP.py</b> there is the from scratch implementation of a Single Layer Perceptron. Different type of learning were implemented: 
- Delta learning: the weights are updated based on the error made before applying the step function.
- Perceptron learning: the weights are updated just when a point is missclassified. 
- Batch learning: all patterns in the training set used as a whole at the same time.
- Sequential learning: step points one by one and update weights successively for each sample.


In order to understand the difference of each learning we plotted the results for two well separated classes from Gaussian distributions. 

In <b>MLP-keras.py</b> a multilayer perceptron of two or three layers was designed and trained in order to make chaotic time-series predictions (regression task). The network has as input 5 nodes of time series data (x(t-20); x(t-15); x(t-10); x(t-5); x(t)) and 1 output node (x(t+5)). The training of this network was done with backpropagation in batch, using early stop, L2 regularization and MSE as error metric. After some hyperparameters tuning it was found out that the best combination was 7 hidden nodes and a regularization value of 0.00000001.

## Conclusions after experiments
SINGLE LAYER PERCEPTRON:
- With perceptron learning the boundary is more near the datapoints that with delta learning. Furthermore, perceptron learning needs some more epochs to converge than delta learning. 
- Sequential learning always a little bit worst than batch learning (in terms of convergence speed).
- If the weights initialization is not optimal the algorithm need more number of epochs to converge. 

MULTI LAYER PERCEPTRON: 
- With a high value of the regularization does not matter the number of hidden nodes because the regularization does not allow the model to become more complex.  

## Libraries needed
- numpy
- pandas
- keras
- matplotlib
