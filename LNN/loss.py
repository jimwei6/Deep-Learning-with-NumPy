import numpy as np

def MSE_LOSS(y_hat, y):
  return ((y_hat - y) ** 2) / 2,  (y_hat - y)/np.prod(y_hat.shape[:-1])