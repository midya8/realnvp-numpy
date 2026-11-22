"""RealNVP affine coupling layers in plain NumPy.

Each coupling keeps one input dimension fixed and pushes the other through
a learned affine map, so the Jacobian is triangular and its log determinant
is just the summed scale. Layers alternate which dimension is fixed.
"""
import numpy as np

from .net import MLP


class Coupling:
    def __init__(self, mask, rng, hidden=64):
        self.mask = np.asarray(mask)
        self.pass_idx = np.where(self.mask == 0)[0]
        self.tr_idx = np.where(self.mask == 1)[0]
        self.net = MLP([len(self.pass_idx), hidden, hidden, 2], rng)
        self._s = None
        self._x2 = None

    def forward(self, x):
        x1 = x[:, self.pass_idx]
        x2 = x[:, self.tr_idx]
        raw = self.net.forward(x1)
        s = np.tanh(raw[:, 0:1])
        t = raw[:, 1:2]
        self._s = s
        self._x2 = x2
        y2 = x2 * np.exp(s) + t
        y = x.copy()
        y[:, self.pass_idx] = x1
        y[:, self.tr_idx] = y2
        return y

    def logdet(self, s):
        return s.sum(axis=1)

    def inverse(self, y):
        y1 = y[:, self.pass_idx]
        y2 = y[:, self.tr_idx]
        raw = self.net.forward(y1)
        s = np.tanh(raw[:, 0:1])
        t = raw[:, 1:2]
        x2 = (y2 - t) * np.exp(-s)
        x = y.copy()
        x[:, self.pass_idx] = y1
        x[:, self.tr_idx] = x2
        return x

    def backward(self, dy, dld):
        """dy: grad wrt this layer's output. dld: grad wrt its logdet term."""
        exp_s = np.exp(self._s)
        dy2 = dy[:, self.tr_idx]
        # dL/ds has two paths, the affine map and this layer's logdet term;
        # s itself is tanh(raw), so the net sees the tanh derivative too
        ds = (dy2 * self._x2 * exp_s + dld) * (1 - self._s ** 2)
        dt = dy2
        dx2 = dy2 * exp_s
        g_in = np.concatenate([ds, dt], axis=1)
        gW, gb, gx1 = self.net.backward(g_in)
        return (gW, gb), dx2, gx1


class RealNVP:
    def __init__(self, n_layers=4, hidden=64, rng=None):
        rng = rng or np.random.default_rng(0)
        masks = [[0, 1], [1, 0]] * ((n_layers + 1) // 2)
        self.layers = [Coupling(masks[i], rng, hidden) for i in range(n_layers)]

    def forward(self, x, keep_scale=False):
        logdet = np.zeros(len(x))
        for layer in self.layers:
            x = layer.forward(x)
            logdet += layer.logdet(layer._s)
        return x, logdet

    def log_prob(self, x, keep_scale=False):
        """log p(x) = log N(f(x)) + log |det df/dx|."""
        z, logdet = self.forward(x, keep_scale)
        base = 0.5 * (z ** 2 + np.log(2 * np.pi)).sum(axis=1)
        return -base + logdet

    def loss_and_grads(self, x):
        """NLL on a batch and gradients for every coupling network."""
        n = len(x)
        z, logdet = self.forward(x, keep_scale=True)
        nll = (0.5 * (z ** 2 + np.log(2 * np.pi)).sum(axis=1) - logdet).mean()
        dz = z / n
        dld = np.full((n, 1), -1.0 / n)
        grads = [None] * len(self.layers)
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            (gW, gb), dx2, gx1 = layer.backward(dz, dld)
            grads[i] = (gW, gb)
            dx = np.zeros_like(x)
            # the pass-through dimension has two paths: identity into the
            # next layer down, and the coupling network
            dx[:, layer.pass_idx] = gx1 + dz[:, layer.pass_idx]
            dx[:, layer.tr_idx] = dx2
            dz = dx
        return nll, grads

    def params(self):
        out = []
        for li, layer in enumerate(self.layers):
            for name, idx in layer.net.params():
                out.append((li, name, idx))
        return out

    def get_flat(self):
        vals = {}
        for li, layer in enumerate(self.layers):
            for k in range(len(layer.net.W)):
                vals[(li, "W", k)] = layer.net.W[k]
                vals[(li, "b", k)] = layer.net.b[k]
        return vals

    def set_flat(self, vals):
        for li, layer in enumerate(self.layers):
            for k in range(len(layer.net.W)):
                layer.net.W[k] = vals[(li, "W", k)]
                layer.net.b[k] = vals[(li, "b", k)]

    def sample(self, n, rng):
        z = rng.normal(size=(n, 2))
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x
