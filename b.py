import torch
from numpy import genfromtxt
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d

# Observed/training input and output
all_x = torch.tensor(genfromtxt('day_length_weight.csv', delimiter=',', usecols=[0, 1])).double().t()
all_y = torch.tensor(genfromtxt('day_length_weight.csv', delimiter=',', usecols=[2])).double().t()

x_train = all_x
y_train = all_y


class LinearRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.randn((1, 2), requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.randn(1, requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W.double() + self.b  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.b, model.W], 0.0001)
for epoch in range(10000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))
