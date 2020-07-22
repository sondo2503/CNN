import numpy as np


def soft_max(x):
    out = np.exp(x)
    return out/np.sum(out)


def categorical_cross_entropy(probs, label):
    return -np.sum(label*np.log(probs))

