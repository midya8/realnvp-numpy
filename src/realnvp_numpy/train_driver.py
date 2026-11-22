"""Train the flow on both datasets, score it, and save the artifacts.

Writes assets/results.json, a sample grid and a density plot per dataset.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from realnvp_numpy.data import make_moons, make_spiral, standardize  # noqa: E402
from realnvp_numpy.flow import RealNVP  # noqa: E402
from realnvp_numpy.metrics import knn_accuracy, mmd_rbf  # noqa: E402
from realnvp_numpy.train import Adam, EMA  # noqa: E402

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets")
STEPS = 12000
BATCH = 256
SEED = 17


def gauss_baseline_nll(train, held):
    mu = train.mean(axis=0)
    var = train.var(axis=0) + 1e-8
    nll = 0.5 * (((held - mu) ** 2) / var + np.log(2 * np.pi * var)).sum(axis=1)
    return float(nll.mean())


def train_dataset(name, gen, seed):
    rng = np.random.default_rng(seed)
    x = gen(2000, rng)
    mu, sd = standardize(x)
    x = (x - mu) / sd
    order = rng.permutation(len(x))
    x = x[order]
    held = x[-600:]
    train = x[:-600]

    model = RealNVP(n_layers=4, hidden=64, rng=np.random.default_rng(seed + 1))
    opt = Adam(model, lr=1e-3)
    ema = EMA(model, decay=0.999)
    curve = []
    for step in range(STEPS):
        idx = rng.integers(0, len(train), BATCH)
        nll, grads = model.loss_and_grads(train[idx])
        opt.step(grads)
        ema.update(model)
        if step % 500 == 0:
            curve.append(float(model.log_prob(train[:600]).mean()))
    model = ema.shadow
    nll_flow = -float(model.log_prob(held).mean())
    nll_gauss = gauss_baseline_nll(train, held)
    samples = model.sample(600, np.random.default_rng(seed + 2))
    knn = knn_accuracy(samples, held)
    mmd = mmd_rbf(samples, held)
    return {
        "steps": STEPS,
        "nll_flow": round(nll_flow, 4),
        "nll_gaussian": round(nll_gauss, 4),
        "knn_acc": round(knn, 4),
        "mmd": round(mmd, 6),
        "train_curve": curve,
    }, model, (mu, sd), held, train


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(ASSETS, exist_ok=True)
    results = {}
    models = {}
    for name, gen, seed in [("moons", make_moons, SEED), ("spiral", make_spiral, SEED + 50)]:
        res, model, (mu, sd), held, train = train_dataset(name, gen, seed)
        results[name] = res
        models[name] = (model, mu, sd)
        rng = np.random.default_rng(seed + 3)

        fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
        data = gen(600, np.random.default_rng(seed + 4))
        data = (data - mu) / sd
        samples = model.sample(600, rng)
        axes[0].scatter(data[:, 0], data[:, 1], s=8, c="0.3")
        axes[0].set_title(f"{name}: data", fontsize=9)
        axes[1].scatter(samples[:, 0], samples[:, 1], s=8, alpha=0.6)
        axes[1].set_title(f"{name}: flow samples", fontsize=9)
        for ax in axes:
            ax.set_xlim(-3.2, 3.2)
            ax.set_ylim(-3.2, 3.2)
        fig.tight_layout()
        fig.savefig(os.path.join(ASSETS, f"samples_{name}.png"), dpi=110)
        plt.close(fig)

        g = np.linspace(-3.2, 3.2, 160)
        gx, gy = np.meshgrid(g, g)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        lp = model.log_prob(pts).reshape(gx.shape)
        fig, ax = plt.subplots(figsize=(4, 3.6))
        ax.contourf(gx, gy, lp, levels=30, cmap="viridis")
        ax.scatter(train[:400, 0], train[:400, 1], s=4, c="white", alpha=0.5)
        ax.set_title(f"{name}: learned density", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(ASSETS, f"density_{name}.png"), dpi=110)
        plt.close(fig)

    results["config"] = {"layers": 4, "hidden": 64, "batch": BATCH, "steps": STEPS, "seed": SEED}
    with open(os.path.join(ASSETS, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: v for k, v in results.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
