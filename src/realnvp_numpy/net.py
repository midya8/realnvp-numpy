"""A small MLP with tanh activations and hand-written backprop.

Each coupling layer uses one of these as its scale and shift network.
The forward pass caches the activations it needs, so the backward pass
needs no computational graph.
"""
import numpy as np


class MLP:
    def __init__(self, dims, rng):
        self.W = []
        self.b = []
        for a, b in zip(dims[:-1], dims[1:]):
            self.W.append(rng.normal(0, np.sqrt(2.0 / a), (a, b)))
            self.b.append(np.zeros(b))
        self._acts = None

    def forward(self, x):
        acts = [x]
        h = x
        for i in range(len(self.W) - 1):
            h = np.tanh(h @ self.W[i] + self.b[i])
            acts.append(h)
        out = h @ self.W[-1] + self.b[-1]
        self._acts = acts
        return out

    def backward(self, grad_out):
        """Weight gradients and the gradient wrt the layer input."""
        acts = self._acts
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        g = grad_out
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = acts[i].T @ g
            gb[i] = g.sum(axis=0)
            if i > 0:
                g = (g @ self.W[i].T) * (1 - acts[i] ** 2)
        # g is the grad wrt the first pre-activation; the input itself has
        # no activation, so one more multiply gets back to it
        return gW, gb, g @ self.W[0].T

    def params(self):
        return [("W", i) for i in range(len(self.W))] + [("b", i) for i in range(len(self.b))]
