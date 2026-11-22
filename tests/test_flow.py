import numpy as np
import pytest

from realnvp_numpy.data import make_moons
from realnvp_numpy.flow import RealNVP


def make_model(seed=3):
    return RealNVP(n_layers=4, hidden=16, rng=np.random.default_rng(seed))


def test_inverse_round_trip():
    model = make_model()
    x = np.random.default_rng(0).normal(size=(50, 2))
    z, _ = model.forward(x)
    back = z
    for layer in reversed(model.layers):
        back = layer.inverse(back)
    assert np.allclose(back, x, atol=1e-8)


def test_logdet_matches_numeric_jacobian():
    model = make_model()
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1, 2))
    eps = 1e-6
    J = np.zeros((2, 2))
    z0, logdet = model.forward(x)
    for j in range(2):
        xp = x.copy(); xp[0, j] += eps
        xm = x.copy(); xm[0, j] -= eps
        J[:, j] = ((model.forward(xp)[0][0] - model.forward(xm)[0][0]) / (2 * eps))
    numeric = np.linalg.slogdet(J)[1]
    assert np.isclose(logdet[0], numeric, atol=1e-4)


def test_masks_alternate():
    model = make_model()
    masks = [tuple(l.mask) for l in model.layers]
    for i in range(1, len(masks)):
        assert masks[i] != masks[i - 1]


def test_loss_decreases():
    model = make_model()
    rng = np.random.default_rng(2)
    data = make_moons(400, rng)
    data = (data - data.mean(0)) / data.std(0)
    first = None
    last = None
    for step in range(60):
        nll, grads = model.loss_and_grads(data[rng.integers(0, len(data), 64)])
        # plain gradient descent is enough for a sanity check
        for li, layer in enumerate(model.layers):
            for k in range(len(layer.net.W)):
                layer.net.W[k] -= 0.01 * grads[li][0][k]
                layer.net.b[k] -= 0.01 * grads[li][1][k]
        if step == 0:
            first = nll
        if step == 59:
            last = nll
    assert last < first


def test_samples_finite_and_shaped():
    model = make_model()
    s = model.sample(100, np.random.default_rng(4))
    assert s.shape == (100, 2)
    assert np.isfinite(s).all()


def test_grad_check_coupling():
    model = make_model()
    rng = np.random.default_rng(5)
    x = rng.normal(size=(32, 2))
    nll, grads = model.loss_and_grads(x)
    li, k = 0, 0
    w = model.layers[li].net.W[k]
    i, j = 0, 0
    eps = 1e-6
    orig = w[i, j]
    w[i, j] = orig + eps
    lp = -model.log_prob(x).mean()
    w[i, j] = orig - eps
    lm = -model.log_prob(x).mean()
    w[i, j] = orig
    num = (lp - lm) / (2 * eps)
    assert np.isclose(num, grads[li][0][k][i, j], atol=1e-4)
