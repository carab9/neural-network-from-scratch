# neural-network-from-scratch
Building a Neural Network from scratch using Numpy.

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
