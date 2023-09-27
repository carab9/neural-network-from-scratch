# neural-network-from-scratch

This project is a to build a Neural Network from scratch using Numpy only (No pytorch or tensorflow). The class NeuralNetwork is a multi-class classifier (2 or more classes). It uses gradient descent for optimization algorithm, Sigmoid for activation function, and Binary Cross Entropy for cost or loss function. It provides APIs to for users to customize a neural network model, including adding layers, and setting user-defined hyperparameters, such as learning rate, number of epochs, regularization rate, random seed, and mesh grid size (for contour drawing). The 8 different test cases are used to show the classication results.

## Architecture

Two major functions in the NeuralNetwork class are fit and predict.

The steps in the fit function are preprare data, initialize neural network (weights, etc), repeat the following step for a number of epochs: loop through all data points and compute cost and gradients, use gradient descent algorithm to update the weights (coefficients) by using the gradients computed. Computing cost and gradients involves forward propagation, compute cost, backward propagation. Regularziation is also factored into the cost and gradient computation.

Predict function uses forward progagation to perform prediction.

## Requirements

Python, numpy, matplotlib.pyplot, random, neural network architecture, machine learning algorithm.

## Technical Skills

Python, numpy, neural network architecture, machine learning algorithm, forward propagation, backward propagation, cost function, activation function, gradient descent algorithm, regularization computation, contour drawing using matplotlib.pyplot, learning rate, and the use of random seed in weight initialization.

## Results

Test case 1: layers 2x5x1 (2 nodes in input layer, 5 nodes in hidden layer, 1 node in output layer)

data = np.array([[-2,-7,0], [3,-3,0], [10,8,0], [16,5,0], [2,2,1], [-1,8,1], [8,10,1], [14,18,1]])

![image](https://github.com/carab9/neural-network-from-scratch/blob/main/neural_network1.png?raw=true)

Test case 2: layers 2x5x5x1 (2 nodes in input layer, 5 nodes in hidden layer 1, 5 nodes in hidden layer 2, 1 node in output layer)

data = np.array([[-2,-7,1], [3,-3,0], [10,8,0], [16,5,1], [2,2,0], [-1,8,1], [8,10,0], [14,18,1]])

![image](https://github.com/carab9/neural-network-from-scratch/blob/main/neural_network2.png?raw=true)

Test case 3: layers 2x5x5x1 (2 nodes in input layer, 5 nodes in hidden layer 1, 5 nodes in hidden layer 2, 1 node in output layer)

data = np.array([[-2,-7,1], [3,-3,0], [10,8,0], [16,5,1], [2,2,0], [-1,8,1], [8,10,0], [9,-7,1], [14,18,1], [5,12,1], [2,-10,1]])

![image](https://github.com/carab9/neural-network-from-scratch/blob/main/neural_network3.png?raw=true)

Test case 4: layers 2x5x5x1 (2 nodes in input layer, 5 nodes in hidden layer 1, 5 nodes in hidden layer 2, 1 node in output layer)

data = np.array([[-2,-7,1],[3,-3,0],[10,8,0],[16,5,1],[2,2,0],[-1,8,1],[8,10,0], [9,-7,1],[14,18,1], [5,12,1],[2,-10,1],[13,10,1],[6,1,1],[5,-10,1]])

![image](https://github.com/carab9/neural-network-from-scratch/blob/main/neural_network4.png?raw=true)

Test case 5: layers 2x8x8x3 (2 nodes in input layer, 8 nodes in hidden layer 1, 8 nodes in hidden layer 2, 3 node in output layer) multi-class classification

data = np.array([[-2,-7,2],[3,-3,0],[10,8,0],[16,5,1],[2,2,0],[-1,8,1],[8,10,0],[9,-7,2],[14,18,1],[5,12,1],[2,-10,2],[13,10,1],[6,1,1],[5,-10,2]])

![image](https://github.com/carab9/neural-network-from-scratch/blob/main/neural_network5.png?raw=true)

Test case 6: layers 2x2x1 (2 nodes in input layer, 2 nodes in hidden layer, 1 node in output layer)

data = np.array([[0,0,1],[1,0,0],[1,1,1],[0,1,0]])

![image](https://github.com/carab9/neural-network-from-scratch/blob/main/neural_network6.png?raw=true)

Test case 7: layers 2x10x1 (2 nodes in input layer, 10 nodes in hidden layer, 1 node in output layer)

data = np.array([[-2,-7,1], [3,-3,0], [10,8,0], [16,5,1], [2,2,0], [-1,8,1], [8,10,0], [14,18,1], [7,0,0],[3,6,0],[6,12,1],[10,-2,1],[3,-6,1],[14,12,1]])

![image](https://github.com/carab9/neural-network-from-scratch/blob/main/neural_network7.png?raw=true)

Test case 8: layers 2x10x1 (2 nodes in input layer, 10 nodes in hidden layer, 1 node in output layer)

data = np.array([[-2,-7,1], [3,-3,0], [10,8,0], [16,5,1], [2,2,0], [-1,8,1], [8,10,0], [14,18,1], [7,0,0],[3,6,0],[6,12,1],[10,-2,1],[3,-6,1],[14,12,1],[6,5,1]])

![image](https://github.com/carab9/neural-network-from-scratch/blob/main/neural_network8.png?raw=true)
