import torch
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import (Enables 3D graphs)
import matplotlib.pyplot as plt

df = pd.read_csv('day_length_weight.csv')

data = torch.tensor(df.values, dtype=torch.float)
all_x = data[:, 1:3].t()
all_y = data[:, 0].t()

# can be changed to make x_test and y_test
x_train = all_x
y_train = all_y

# inputs to our model
n = 2


class LinearRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.randn((1, n), requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.randn(1, requires_grad=True)

    # Predictor
    def f(self, x):
        return self.W.mm(x) + self.b

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y.reshape(1, 1000))


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.0001)
for epoch in range(75000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

fig = plt.figure().gca(projection='3d')

fig.scatter(df['length'], df['weight'], df['day'], c='red')

fig.scatter(x_train[0, :], x_train[1, :], zs=model.f(x_train).detach(), c='blue')

fig.set_xlabel('Day')
fig.set_ylabel('Length')
fig.set_zlabel('Weight')

plt.savefig('b')
