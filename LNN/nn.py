class Module:
  def __init__(self):
    self.layers = []
    self.params = []

  def add_layers(self, layer):
    self.layers.append(layer)
    if layer.params:
      for i in range(len(layer.params)):
        self.params.append([layer.params[i], layer.grads[i]])

  def forward(self, X):
    self.X = X
    for layer in self.layers:
      X = layer.forward(X)
    return X
  
  def backward(self, loss_grad):
    for i in range(len(self.layers) - 1, -1, -1):
      loss_grad = self.layers[i].backward(loss_grad)
    return loss_grad