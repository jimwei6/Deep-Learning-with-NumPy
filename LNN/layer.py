# Custom Deep Learning Layers with Numpy
import numpy as np
from .validator import numerical_gradient_from_df

# An example is each row
class Layer:
  def __init__(self):
    self.params = None
    pass

  def forward(self, X):
    return NotImplementedError

  def backward(self, grad):
    return NotImplementedError
  
class Linear(Layer):
  def __init__(self, input_dim, out_dim):
    super().__init__()
    self.id = id
    self.W = np.random.randn(input_dim, out_dim) * 0.01
    self.b = np.zeros(out_dim)
    self.params = [self.W, self.b]
    self.grads = [np.zeros_like(self.W), np.zeros_like(self.b)]

  def forward(self, X):
    self.X = X
    self.O = X @ self.W + self.b
    return self.O

  def backward(self, grad):
    self.dW = np.dot(self.X.T, grad) / self.X.shape[0]
    self.db = np.sum(grad, axis=0) / self.X.shape[0]
    self.grads[0] += self.dW
    self.grads[1] += self.db
    return np.dot(grad, self.W.T)
    
  


