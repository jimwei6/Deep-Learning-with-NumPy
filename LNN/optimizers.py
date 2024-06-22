import numpy as np

class Optimizer():
  def __init__(self):
    return NotImplementedError

  def step(self):
    return NotImplementedError

  def zero_grad(self):
    return NotImplementedError
  
class SGD(Optimizer):
  def __init__(self, model_params, lr=0.01, momentum=0.4):
    self.lr = lr
    self.params = model_params
    self.momentum = momentum
    self.vs = [] # store
    for p, grad in self.params:
      self.vs.append(np.zeros_like(p))

  def zero_grad(self):
    for i, _ in enumerate(self.params):
      self.params[i][1].fill(0) # set all gradients to 0

  def step(self):
    for i, _ in enumerate(self.params):
      p, grad = self.params[i]
      self.vs[i] = self.momentum * self.vs[i] + self.lr * grad # calculate 
      self.params[i][0] -= self.vs[i]

class Adam(Optimizer):
  def __init__(self, model_params, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
    self.lr = lr
    self.params = model_params
    self.b1, self.b2, self.eps = b1, b2, eps
    self.ms = []
    self.vs = []
    self.t = 0
    for p, grad in self.params:
      self.vs.append(np.zeros_like(p))
      self.ms.append(np.zeros_like(p))

  def zero_grad(self):
    for i, _ in enumerate(self.params):
      self.params[i][1].fill(0) # set all gradients to 0

  def step(self):
    self.t += 1
    for i, _ in enumerate(self.params):
      p, grad = self.params[i]
      
      self.vs[i] = self.b2 * self.vs[i] + (1-self.b2) * (grad ** 2)
      self.ms[i] = self.b1 * self.ms[i] + (1-self.b1) * grad

      m1 = self.ms[i]/(1 - np.power(self.b1, self.t))
      v1 = self.vs[i]/(1 - np.power(self.b2, self.t))

      self.params[i][0] -= self.lr * m1 / (np.sqrt(v1) + self.eps)