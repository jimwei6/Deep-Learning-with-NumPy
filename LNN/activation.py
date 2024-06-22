import numpy as np
from .layer import Layer

class Sigmoid(Layer):
  def __init__(self):
    super().__init__()
    pass

  def forward(self, x):
    self.x = x
    return 1.0 / (1 + np.exp(-x))
  
  def backward(self, grad):
    s = 1.0 / (1 + np.exp(-self.x))
    return grad * s * (1 - s)
  
class Tanh(Layer):
  def __init__(self):
    super().__init__()
    pass

  def forward(self, x):
    self.x = x
    return np.tanh(x)
  
  def backward(self, grad):
    return grad * (1 - np.square(np.tanh(self.x)))
  
class ReLu(Layer):
  def __init__(self):
    super().__init__()
    pass

  def forward(self, x):
    self.x = x
    return np.maximum(x, 0)
  
  def backward(self, grad):
    return grad * (self.x > 0)