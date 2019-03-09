import numpy as np


# activate function
def sigmoid(z):
    return 1/(1 + np.exp(-1*z))


# derivation of sigmoid
def sig_deriv(z):
    return np.exp(-1*z) / ((1 + np.exp(-1*z)) ** 2)


# softmax
def softmax(z):
    z = np.exp(z)
    d = np.sum(z, axis=1)
    return z/d[:, np.newaxis]


def sm_deriv(a):
    return a*(1-a)


def relu(z):
    z[z <= 0] = 0
    return z
