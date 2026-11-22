"""Adam training loop for the flow, with a weight EMA."""
import copy

import numpy as np


class Adam:
    def __init__(self, model, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.model = model
        self.lr = lr
        self.b1, self.b2, self.eps = b1, b2, eps
        self.m = {}
        self.v = {}
        self.t = 0
        for key in self._keys():
            self.m[key] = 0.0
            self.v[key] = 0.0

    def _keys(self):
        out = []
        for li, layer in enumerate(self.model.layers):
            for k in range(len(layer.net.W)):
                out.append((li, k, "W"))
                out.append((li, k, "b"))
        return out

    def step(self, grads):
        self.t += 1
        c1 = 1 - self.b1 ** self.t
        c2 = 1 - self.b2 ** self.t
        for li, layer in enumerate(self.model.layers):
            gW, gb = grads[li]
            for k in range(len(layer.net.W)):
                for name, g, get, set_ in [
                    ("W", gW[k], lambda: layer.net.W[k], None),
                    ("b", gb[k], lambda: layer.net.b[k], None),
                ]:
                    key = (li, k, name)
                    self.m[key] = self.b1 * self.m[key] + (1 - self.b1) * g
                    self.v[key] = self.b2 * self.v[key] + (1 - self.b2) * g ** 2
                    update = self.lr * (self.m[key] / c1) / (np.sqrt(self.v[key] / c2) + self.eps)
                    if name == "W":
                        layer.net.W[k] -= update
                    else:
                        layer.net.b[k] -= update


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)

    def update(self, model):
        for li, layer in enumerate(model.layers):
            for k in range(len(layer.net.W)):
                e_layer = self.shadow.layers[li]
                e_layer.net.W[k] = self.decay * e_layer.net.W[k] + (1 - self.decay) * layer.net.W[k]
                e_layer.net.b[k] = self.decay * e_layer.net.b[k] + (1 - self.decay) * layer.net.b[k]
