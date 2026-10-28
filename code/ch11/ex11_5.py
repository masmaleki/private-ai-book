"""Exercise 11-5: add tanh and exp to Value and gradient-check them."""
import math
from micrograd import Value


def v_tanh(self):
    t = math.tanh(self.data)
    out = Value(t, (self,), "tanh")
    def _backward():
        self.grad += (1 - t * t) * out.grad     # d tanh(x)/dx = 1 - tanh^2
    out._backward = _backward
    return out


def v_exp(self):
    e = math.exp(self.data)
    out = Value(e, (self,), "exp")
    def _backward():
        self.grad += e * out.grad               # d exp(x)/dx = exp(x)
    out._backward = _backward
    return out


Value.tanh, Value.exp = v_tanh, v_exp

for name, f in [("tanh", math.tanh), ("exp", math.exp)]:
    x = Value(0.7)
    y = getattr(x, name)()
    y.backward()
    eps = 1e-6
    numeric = (f(0.7 + eps) - f(0.7 - eps)) / (2 * eps)
    print(f"{name}: autograd={x.grad:.6f} numeric={numeric:.6f}")

# Exercise 11-3 check
from micrograd import MLP
net = MLP([2, 2, 1])
(h1, h2), (o,) = net.layers
h1.w[0].data, h1.w[1].data, h1.b.data = 0.5, -0.2, 0.1
h2.w[0].data, h2.w[1].data, h2.b.data = 0.3, 0.8, -0.1
o.w[0].data, o.w[1].data, o.b.data = 0.7, -0.4, 0.2
loss = (net([1.0, -1.0]) - 0.2) ** 2
loss.backward()
print("ex 11-3 loss", round(loss.data, 4), [round(p.grad, 4) for p in net.parameters()])
