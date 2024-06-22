import numpy as np

def numerical_gradient_from_df(f, p, df, h=1e-5):
  grad = np.zeros_like(p)
  it = np.nditer(p, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:
    idx = it.multi_index

    oldval = p[idx]
    p[idx] = oldval + h
    pos = f()

    p[idx] = oldval - h
    neg = f()
    p[idx] = oldval
    grad[idx] = np.sum((pos - neg) * df) / (2 * h)
    it.iternext()
  return grad

def verify_grad(layer):
  x = np.random.randn(3,48)  
  layer.forward(x)
  df = np.random.randn(3, 10)
  dx = layer.backward(df)
  dx_num = numerical_gradient_from_df(lambda: layer.forward(x), x, df)
  diff_error = lambda x,y: np.max(np.abs(x-y))
  print(diff_error(dx, dx_num))