import LNN.activation
import LNN.loss
import LNN.nn
import numpy as np

from sklearn import datasets

import LNN.layer
import LNN.optimizers

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

model = LNN.nn.Module()
model.add_layers(LNN.layer.Linear(diabetes_X[0].shape[0], 100))
model.add_layers(LNN.activation.ReLu())
model.add_layers(LNN.layer.Linear(100, 1))
model.add_layers(LNN.activation.ReLu())
optimizer = LNN.optimizers.Adam(model.params)

for e in range(1000):
  losses = []
  for i in range(len(diabetes_X_train)//5):
    optimizer.zero_grad()
    f = model.forward(diabetes_X_train[i:i+5])
    loss, loss_grad = LNN.loss.MSE_LOSS(f, diabetes_y_train[i:i+5].reshape((5, 1)))
    model.backward(loss_grad)
    optimizer.step()
    losses.append(loss)


for i in range(len(diabetes_X_test)):
  losses = []
  f = model.forward(diabetes_X_test[i:i+1])
  loss, loss_grad = LNN.loss.MSE_LOSS(f, diabetes_y_test[i:i+1].reshape((1, 1)))
  losses.append(loss)

print(np.mean(losses))