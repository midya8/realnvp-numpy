"""Toy 2D datasets and standardization."""
import numpy as np


def make_moons(n, rng):
    n1 = n // 2
    n2 = n - n1
    t1 = rng.uniform(0, np.pi, n1)
    t2 = rng.uniform(0, np.pi, n2)
    x1 = np.stack([np.cos(t1), np.sin(t1)], axis=1)
    x2 = np.stack([1 - np.cos(t2), 1 - np.sin(t2) - 0.5], axis=1)
    x = np.concatenate([x1, x2], axis=0)
    x += rng.normal(0, 0.08, x.shape)
    return x


def make_spiral(n, rng):
    t = np.sqrt(rng.uniform(0, 1, n)) * 3 * np.pi
    r = t / (3 * np.pi)
    x = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    x += rng.normal(0, 0.05, x.shape)
    return x


def standardize(x):
    return x.mean(axis=0), x.std(axis=0)
