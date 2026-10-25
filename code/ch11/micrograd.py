"""micrograd.py - a tiny scalar autograd engine (teaching only, ~80 lines of logic).
Inspired by the idea of micrograd: every number remembers how it was made,
so we can walk backwards and compute gradients with the chain rule.
"""
import math
import random


class Value:
    def __init__(self, data, parents=(), op=""):
        self.data = float(data)
        self.grad = 0.0
        self._parents = parents
        self._backward = lambda: None
        self.op = op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad          # d(a+b)/da = 1
            other.grad += out.grad         # d(a+b)/db = 1
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad   # d(a*b)/da = b
            other.grad += self.data * out.grad   # d(a*b)/db = a
        out._backward = _backward
        return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self,), f"**{k}")
        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")
        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    __radd__ = __add__
    __rmul__ = __mul__

    def backward(self):
        # topological order: every node after all nodes it depends on
        order, seen = [], set()
        def visit(v):
            if v not in seen:
                seen.add(v)
                for p in v._parents:
                    visit(p)
                order.append(v)
        visit(self)
        self.grad = 1.0
        for v in reversed(order):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


class Neuron:
    def __init__(self, n_in, relu=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.relu = relu

    def __call__(self, x):
        z = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return z.relu() if self.relu else z

    def parameters(self):
        return self.w + [self.b]


class MLP:
    def __init__(self, sizes):
        # sizes like [2, 2, 1]: last layer is linear (no ReLU)
        self.layers = [[Neuron(a, relu=(i < len(sizes) - 2)) for _ in range(b)]
                       for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]))]

    def __call__(self, x):
        for layer in self.layers:
            x = [n(x) for n in layer]
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for n in layer for p in n.parameters()]


def hand_example():
    """The 2-2-1 network computed by hand in the chapter."""
    net = MLP([2, 2, 1])
    h1, h2 = net.layers[0]
    out = net.layers[1][0]
    h1.w[0].data, h1.w[1].data, h1.b.data = 0.5, -0.2, 0.1
    h2.w[0].data, h2.w[1].data, h2.b.data = 0.3, 0.8, -0.1
    out.w[0].data, out.w[1].data, out.b.data = 0.7, -0.4, 0.2
    x, target, lr = [1.0, 0.5], 0.5, 0.1
    for step in range(2):
        y = net(x)
        loss = (y - target) ** 2
        for p in net.parameters():
            p.grad = 0.0
        loss.backward()
        print(f"step {step}: y_hat={y.data:.5f} loss={loss.data:.5f}")
        if step == 0:
            names = ["w11", "w12", "b1", "w21", "w22", "b2", "v1", "v2", "c"]
            print("  grads:", {n: round(p.grad, 4) for n, p in zip(names, net.parameters())})
        for p in net.parameters():
            p.data -= lr * p.grad


def train_toy():
    """Fit a tiny MLP to synthetic price-vs-quantity points (example data)."""
    random.seed(0)
    # x = log10(qty) - 3, y = unit price in EUR (synthetic, decreasing with qty)
    data = [(q, 1.2 - 0.25 * q + 0.08 * q * q) for q in [-1, -0.5, 0, 0.5, 1, 1.3]]
    net = MLP([1, 8, 1])
    for epoch in range(301):
        loss = sum(((net([x]) - y) ** 2 for x, y in data), Value(0.0)) * (1 / len(data))
        for p in net.parameters():
            p.grad = 0.0
        loss.backward()
        for p in net.parameters():
            p.data -= 0.05 * p.grad
        if epoch % 100 == 0:
            print(f"epoch {epoch:3d}  mse={loss.data:.5f}")
    for x, y in data:
        print(f"  x={x:+.1f} target={y:.3f} pred={net([x]).data:.3f}")


if __name__ == "__main__":
    hand_example()
    train_toy()
