import numpy as np
import matplotlib.pyplot as plt
import random

# class NeuralNetwork
# a multi-class classifier
# optimization algorithm: gradient descent
# activation function: Sigmoid
# cost or loss function: binary cross entropy
class NeuralNetwork:
    # constructor
    def __init__(self):
      # data
      self.data = None
      self.X = None
      self.y = None

      # input layer, hidden layer(s), and output layer
      self.layers = []

      # coefficients (or weights)
      self.thetas = []

      # user-defined hyperparameters
      # (learning parameters)

      # learning rate
      self.alpha = 0.1

      # random seed
      # seed=27,32,33,35,38,39
      self.seed = 27

      # number of iterations in training
      self.num_iter = 10000

      # regularization rate
      self.rlambda = 0

      # mesh grid size
      self.grid_size = 1000

    # set learning rate
    def set_learning_rate(self, alpha):
      self.alpha = alpha

    # set random seed
    def set_random_seed(self, seed):
      self.seed = seed

    # set number of iterations in training
    def set_num_iterations(self, num_iter):
      self.num_iter = num_iter

    # set regularization rate
    def set_reg_rate(self, rlambda):
      self.rlambda = rlambda

    # set grid size
    def set_grid_size(self, grid_size):
      self.grid_size = grid_size

    # add a layer to the end
    def add_layer(self, num_neurons):
      self.layers.append(num_neurons)

    @staticmethod
    def one_encode(i, size):
      encoded = np.zeros((1, size))
      encoded[0, i % size] = 1.0
      return encoded

    @staticmethod
    def sigmoid(z):
      return (1.0/(1+np.exp(0-z)))

    # delta for output layer
    @staticmethod
    def delta_output_layer(ai, yi):
      # single data point
      # delta_layer = ai_layer - yi
      delta = ai - yi
      return delta

    # delta for hidden layer
    @staticmethod
    def delta_hidden_layer(theta, delta_next, ai):
      # single data point
      # delta = thetalayer.T * delta_layer+1 .* ai_layer .* (1 - ai_layer)
      delta = np.multiply(np.dot(np.transpose(theta), delta_next), np.multiply(ai, (1 - ai)))
      # drop delta0
      delta = delta[1:]
      return delta

    # gradient descent
    @staticmethod
    def gradient_descent(delta_next, ai):
      # single data point
      # grad_layer = delta_layer+1 * ai_layer.T
      grad = np.dot(delta_next, np.transpose(ai))
      return grad

    # forward propagation
    def forward_propagation(self, xi):
      # single data point
      al = [xi]
      ai = xi
      for i in range(len(self.thetas)):
        theta = self.thetas[i]

        # zi = theta * (ai)
        # ai = exp(-zi)
        zi = np.dot(theta, ai)
        ai1 = self.sigmoid(zi)

        # do not add bias unit to last ai
        if i != len(self.thetas)-1:
          o = np.ones((1, 1))
          ai1 = np.append(o, ai1, axis=0)

        al.append(ai1)
        ai = ai1
      return al

    # back propagation
    # compute cost and gradients for a single data point
    def back_propagation(self, xi, yi):
      # init gradient matrices
      grad = []
      for i in range(len(self.thetas)):
        grad.append(np.zeros(self.thetas[i].shape))

      # forward propagation
      al = self.forward_propagation(xi)
      #print("al", al)

      # cost function
      i = len(al)-1
      #print("i", i)
      ai = al[i]
      J = np.sum(np.multiply(yi, (0-np.log(ai))) +
                 np.multiply((1 - yi), (0-np.log(1 - ai))))

      # back propagation
      # gradient descent
      # output layer
      delta = self.delta_output_layer(ai, yi)
      grad[i-1] = self.gradient_descent(delta, al[i-1])

      # hidden layer
      prev = delta
      for i in range(len(al)-2, 0, -1):
        #print("i", i)
        ai = al[i]
        delta = self.delta_hidden_layer(self.thetas[i], prev, ai)
        grad[i-1] = self.gradient_descent(delta, al[i-1])
        prev = delta

      return [J, grad]

    # compute cost and gradient
    # (with regularization for training)
    def cost_and_gradient(self):
      # cost function
      # add y * (-log (h)) + (1 - y) * (-log(1 - h))
      # for all output layers for every data points
      J = 0.0
      # init theta gradient matrices
      theta_grad = []
      for i in range(len(self.thetas)):
        theta_grad.append(np.zeros(self.thetas[i].shape))

      # loop through each data point and accumulate gradient
      # for each layer
      m = self.X.shape[0]
      for i in range(0, m):
        #print(i)
        xi = np.matrix(self.X[i])
        xi = np.transpose(xi)
        yi = np.matrix(self.y[i])
        yi = np.transpose(yi)
        #print("xi", xi)
        #print("yi", yi)

        # compute cost and gradients for a single data point
        [cost, grad] = self.back_propagation(xi, yi)
        #print(cost)
        #print(grad)
        J += cost
        for i in range(len(theta_grad)):
          theta_grad[i] = theta_grad[i] + grad[i]

      # regularization (cost function)
      # add theta[i,j]**2 for all layers (j>0) to the cost
      R = 0.0
      if self.rlambda > 0:
        for i in range(len(self.thetas)):
          theta = np.copy(self.thetas[i])
          theta[:,0] = 0
          R += np.sum(np.multiply(theta, theta))
      J = J * 1.0/m + self.rlambda/(2*m) * R

      for i in range(len(theta_grad)):
        # regularization (gradient descent)
        # add lambda * theta[i, j] (j>0) to theta_grad[l]
        if self.rlambda > 0:
          theta = np.copy(self.thetas[i])
          theta[:,0] = 0
          theta_grad[i] = theta_grad[i] + self.rlambda * theta
        theta_grad[i] = 1.0/m * theta_grad[i]

      return [J, theta_grad]

    # compute averaged cost for all data points
    # (with or without regularization)
    def cost_function(self, X, y, reg=False):
      # add y * (-log (h)) + (1 - y) * (-log(1 - h))
      # for all output layers for every data points
      J = 0.0
      m = X.shape[0]

      # loop through each data point and accumulate cost
      for i in range(0, m):
        #print(i)
        xi = np.matrix(X[i])
        xi = np.transpose(xi)
        yi = np.matrix(y[i])
        yi = np.transpose(yi)
        #print("xi", xi)
        #print("yi", yi)

        # forward propagation
        al = self.forward_propagation(xi)
        ai = al[len(al)-1]
        J += np.sum(np.multiply(yi, (0-np.log(ai))) +
                    np.multiply((1 - yi), (0-np.log(1 - ai))))

      # regularization (cost function)
      # add theta[i,j]**2 for all layers (j>0) to the cost
      R = 0.0
      if reg and self.rlambda > 0:
        for i in range(len(self.thetas)):
          theta = np.copy(self.thetas[i])
          theta[:,0] = 0
          R += np.sum(np.multiply(theta, theta))

      J = 1.0/m * J + self.rlambda/(2*m) * R
      return J

    # init thetas (coefficients)
    def init_thetas(self):
      np.random.seed(self.seed)
      for i in range(len(self.layers)-1):
        self.thetas.append(np.random.randn(self.layers[i+1], self.layers[i]+1))

      #for i in range(len(self.thetas)):
      #  print("theta", i + 1)
      #  print(self.thetas[i])

    # init data for input layer
    def init_input_data(self):
      data = np.copy(self.data)

      # randomly shuffle the training data first
      np.random.shuffle(data)
      #print("random", data)

      n = data.shape[1]
      #print(n)

      # X, y
      # row matrix
      #x = data[:, [0, n-2]]
      x = data[:, 0: n-1]
      y = data[:, [n-1]]
      #print(x)
      #print(y)
      #print(x.shape)
      #print(y.shape)

      # column matrix
      o = np.ones((x.shape[0], 1))
      #print(o)
      X = np.append(o, x, axis=1)

      self.X = X
      self.y = y
      # multi-class classification
      # convert y to one and zeros column
      num_output = self.layers[len(self.layers)-1]
      if num_output >= 3:
        tmp = self.one_encode(y[0,0], num_output)
        for i in range(1, y.shape[0]):
          tmp = np.concatenate((tmp,self.one_encode(y[i,0], num_output)), axis=0)
        self.y = tmp

      #print("X", self.X)
      #print("y", self.y)
      #print(self.y.shape)

    # train a neural network
    def fit(self, data):
      self.data = data
      print("data", data)

      # init thetas (coefficients)
      self.init_thetas()

      count = 0
      while count < self.num_iter:
        count += 1

        # randomly shuffle the training data and
        # init data for input layer
        self.init_input_data()

        # loop through all data points and compute
        # averaged cost and gradient descents
        [J, theta_grad] = self.cost_and_gradient()
        #print("cost", J)

        # update thetas (coefficients)
        for i in range(len(self.thetas)):
           self.thetas[i] = self.thetas[i] - self.alpha * theta_grad[i]

      print("Gradient Decent parameters:")
      for i in range(len(self.thetas)):
        print(self.thetas[i])

      # compute cost
      #print(self.X)
      #print(self.y)
      J = self.cost_function(self.X, self.y)
      print("cost", J)

    # predict a data
    def predict(self, x_pred):
      for x in x_pred:
        print("prediction", x)
        xi = np.matrix(x)
        xi = np.transpose(xi)
        o = np.ones((1, 1))
        xi = np.append(o, xi, axis=0)
        al = self.forward_propagation(xi)
        ai_pred = al[len(al)-1]
        #print(ai_pred)
        if len(ai_pred) == 1:
          # binary classification
          if ai_pred[0,0] >= 0.5:
            print("label 1")
          else:
            print("label 0")
        else:
          # multi-class classification
          print("label", np.argmax(ai_pred, axis=0)[0,0])

    # hypothesis mesh for countur
    def hypothesis_mesh(self, X1, X2):
      #print("X1.shape={}".format(X1.shape))
      #print("X2.shape={}".format(X2.shape))

      # hidden layer 1
      (dim_x, dim_y) = X1.shape
      n = self.thetas[0].shape[0]
      Z = np.zeros((n, dim_x, dim_y))
      a = np.zeros((n+1, dim_x, dim_y))
      #print("Z.shape={}".format(Z.shape))
      #print("a.shape={}".format(a.shape))

      for i in range(n):
        Z[i] = self.thetas[0].item((i, 0)) + self.thetas[0].item((i, 1)) * X1 + self.thetas[0].item((i, 2)) * X2
      a[0,:,:] = 1
      a[1:,:,:] = self.sigmoid(Z)
      a2 = a

      # hidden layer 2+
      for l in range(1, len(self.thetas)-1):
        n = self.thetas[l].shape[0]
        Z = np.zeros((n, dim_x, dim_y))
        a = np.zeros((n+1, dim_x, dim_y))
        #print("Z.shape={}".format(Z.shape))
        #print("a.shape={}".format(a.shape))
        for i in range(n):
          n2 = self.thetas[l].shape[1]
          for j in range(n2):
            Z[i] += self.thetas[l].item((i, j))*a2[j,:,:]
        a[0,:,:] = 1
        a[1:,:,:] = self.sigmoid(Z)
        a2 = a

      # output layer
      l = len(self.thetas) - 1
      n = self.thetas[l].shape[0]
      Z = np.zeros((n, dim_x,dim_y))
      a = np.zeros((n, dim_x, dim_y))
      #print("Z.shape={}".format(Z.shape))
      #print("a.shape={}".format(a.shape))
      for i in range(n):
        n2 = self.thetas[l].shape[1]
        for j in range(n2):
          Z[i] += self.thetas[l].item((i, j))*a2[j,:,:]
      a = self.sigmoid(Z)
      return a

    # display contour
    def display_contour(self, grid_size =1000):
      self.grid_size = grid_size

      colors=['blue', 'red', 'green', 'orange','purple',
              'brown', 'pink', 'gray', 'olive', 'cyan']
      fig,ax=plt.subplots(1,1)

      # plot data
      # num of neurons in the output layer
      num_output = self.layers[len(self.layers)-1]
      if num_output >= 3:
        # multi-class classification
        for i in range(num_output):
          x0 = np.array(self.data[:, 0][self.data[:, 2] == i])
          y0 = np.array(self.data[:, 1][self.data[:, 2] == i])
          ax.scatter(x0, y0, color=colors[i])
      else:
        # binary classification
        x0 = np.array(self.data[:, 0][self.data[:, 2] == 0])
        y0 = np.array(self.data[:, 1][self.data[:, 2] == 0])
        x1 = np.array(self.data[:, 0][self.data[:, 2] == 1])
        y1 = np.array(self.data[:, 1][self.data[:, 2] == 1])
        ax.scatter(x0, y0, color=colors[0])
        ax.scatter(x1, y1, color=colors[1])

      # plot contour
      x_min = np.min(self.data[:, 0]) - 4
      x_max = np.max(self.data[:, 0]) + 4
      y_min = np.min(self.data[:, 1]) - 4
      y_max = np.max(self.data[:, 1]) + 4

      # create mesh grid matrices
      xlist = np.linspace(x_min, x_max, self.grid_size)
      ylist = np.linspace(y_min, y_max, self.grid_size)
      X, Y = np.meshgrid(xlist, ylist)

      # plot contour
      Z = self.hypothesis_mesh(X, Y);
      #print(Z)

      labels = [0.5]
      for i in range(len(Z)):
        cp = ax.contour(X, Y, Z[i], labels)
        ax.clabel(cp, inline=True, fontsize=10, colors=colors[i])

# example 1
# x0, x1, x2
# 2x5x1
data = np.array([[-2,-7,0], [3,-3,0], [10,8,0], [16,5,0], [2,2,1], [-1,8,1], [8,10,1], [14,18,1]])
nn = NeuralNetwork()
nn.add_layer(2)
nn.add_layer(5)
nn.add_layer(1)
nn.fit(data)
x_pred = np.array([[10, 2], [5, 10]])
nn.predict(x_pred)
nn.display_contour()

# example 2
# x0, x1, x2
# 2x5x5x1
data = np.array([[-2,-7,1], [3,-3,0], [10,8,0], [16,5,1], [2,2,0], [-1,8,1], [8,10,0], [14,18,1]])
nn = NeuralNetwork()
nn.add_layer(2)
nn.add_layer(5)
nn.add_layer(5)
nn.add_layer(1)
nn.fit(data)
x_pred = np.array([[10, 2], [15, 0]])
nn.predict(x_pred)
nn.display_contour()

# example 3
# x0, x1, x2
# 2x5x5x1
data = np.array([[-2,-7,1], [3,-3,0], [10,8,0], [16,5,1], [2,2,0], [-1,8,1], [8,10,0], [9,-7,1], [14,18,1], [5,12,1], [2,-10,1]])
nn = NeuralNetwork()
nn.add_layer(2)
nn.add_layer(5)
nn.add_layer(5)
nn.add_layer(1)
#nn.set_reg_rate(0.001)
nn.fit(data)
x_pred = np.array([[10, 2], [5, 0]])
nn.predict(x_pred)
nn.display_contour()

# example 4
# x0, x1, x2
# 2x5x5x1
data = np.array([[-2,-7,1],[3,-3,0],[10,8,0],[16,5,1],[2,2,0],[-1,8,1],[8,10,0], [9,-7,1],[14,18,1], [5,12,1],[2,-10,1],[13,10,1],[6,1,1],[5,-10,1]])
nn = NeuralNetwork()
nn.add_layer(2)
nn.add_layer(5)
nn.add_layer(5)
nn.add_layer(1)
nn.set_random_seed(39)
nn.fit(data)
x_pred = np.array([[10, 2], [5, 3]])
nn.predict(x_pred)
nn.display_contour()

# example 5
# x0, x1, x2
# 2x8x8x3
# multi-class classification
data = np.array([[-2,-7,2],[3,-3,0],[10,8,0],[16,5,1],[2,2,0],[-1,8,1],[8,10,0],[9,-7,2],[14,18,1],[5,12,1],[2,-10,2],[13,10,1],[6,1,1],[5,-10,2]])
nn = NeuralNetwork()
nn.add_layer(2)
nn.add_layer(8)
nn.add_layer(8)
nn.add_layer(3)
nn.set_random_seed(27)
nn.fit(data)
x_pred = np.array([[10, 2], [0, -7]])
nn.predict(x_pred)
nn.display_contour()

# example 6
# x0, x1, x2
# 2x2x1
data = np.array([[0,0,1],[1,0,0],[1,1,1],[0,1,0]])
nn = NeuralNetwork()
nn.add_layer(2)
nn.add_layer(2)
nn.add_layer(1)
nn.set_random_seed(39)
nn.fit(data)
# predict (10, 2), (1, 12)
x_pred = np.array([[10, 2], [1, 12]])
nn.predict(x_pred)
nn.display_contour()

# example 7
# x0, x1, x2
# 2x10x1
data = np.array([[-2,-7,1], [3,-3,0], [10,8,0], [16,5,1], [2,2,0], [-1,8,1], [8,10,0], [14,18,1], [7,0,0],[3,6,0],[6,12,1],[10,-2,1],[3,-6,1],[14,12,1]])
nn = NeuralNetwork()
nn.add_layer(2)
nn.add_layer(10)
nn.add_layer(1)
nn.fit(data)
x_pred = np.array([[10, 2], [1, 12]])
nn.predict(x_pred)
nn.display_contour()

# example 8
# x0, x1, x2
# 2x10x1
data = np.array([[-2,-7,1], [3,-3,0], [10,8,0], [16,5,1], [2,2,0], [-1,8,1], [8,10,0], [14,18,1], [7,0,0],[3,6,0],[6,12,1],[10,-2,1],[3,-6,1],[14,12,1],[6,5,1]])
nn = NeuralNetwork()
nn.add_layer(2)
nn.add_layer(10)
nn.add_layer(1)
nn.set_random_seed(32)
nn.fit(data)
x_pred = np.array([[10, 2], [1, 12]])
nn.predict(x_pred)
nn.display_contour()