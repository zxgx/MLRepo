from keras.datasets import mnist
import keras.utils as utils
import numpy as np
import utils as funcs
import matplotlib.pylab as plt

# Preprocess
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2])).astype(np.float64) / 255
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2])).astype(np.float64) / 255
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

# Network construction
neurons = (x_train.shape[1], 500, 500, 10)
layers = len(neurons)-1
eta = 0.001
W, B, L = [], [], []
ww, bb = [], []
for layer in range(layers):
    W.append(np.random.rand(neurons[layer], neurons[layer+1]))
    ww.append(np.zeros((neurons[layer], neurons[layer+1])))
    B.append(np.random.rand(1, neurons[layer+1]))
    bb.append(np.zeros((1, neurons[layer+1])))

# Training
iterations, batchs = 20, 100
for i in range(iterations):
    for batch in range(x_train.shape[0]//batchs):
        X = x_train[batch*batchs:batchs*(batch+1), :]
        a = []

        # forward pass
        for layer in range(layers):
            z = X.dot(W[layer]) + B[layer]
            if layer != layers-1:
                a.append(funcs.relu(z))
                # a.append(funcs.sigmoid(z))
            else:
                a.append(funcs.softmax(z))
            X = a[-1]

        # recording error, the crossentropy
        Y = y_train[batch*batchs:(batch+1)*batchs, :]
        ln = np.sum(-1 * np.log(X) * Y, axis=1)
        L.append(sum(ln))

        # backward pass
        plpa = -1.0 * Y / X
        for layer in range(layers-1, -1, -1):
            # papz = a[layer] * (1 - a[layer])
            papz = np.ones(a[layer].shape)
            if layer == 0:
                pzpw = x_train[batch*batchs:(batch+1)*batchs, :]
            else:
                pzpw = a[layer-1]
            plpz = plpa * papz

            plpa = plpz.dot(W[layer].T)

            # Vanilla gradient descent
            for col in range(plpz.shape[1]):
                plpw = np.sum((plpz[:, col][:, np.newaxis] * pzpw), axis=0).T
                ww[layer][:, col] += plpw ** 2
                W[layer][:, col] = W[layer][:, col] - eta * plpw / np.sqrt(ww[layer][:, col])
            bb[layer] += np.sum(plpz, axis=0) ** 2
            B[layer] = B[layer] - eta * np.sum(plpz, axis=0) / np.sqrt(bb[layer])


# validation on training data
X = x_train
Y = x_test
for layer in range(layers):
    X = X.dot(W[layer])+B[layer]
    Y = Y.dot(W[layer])+B[layer]
    if layer == layers-1:
        X = funcs.softmax(X)
        Y = funcs.softmax(Y)
    else:
        X = funcs.sigmoid(X)
        Y = funcs.sigmoid(Y)

res = np.argmax(X, axis=1)
trueth = np.argmax(y_train, axis=1)
accuracy = np.sum(res == trueth) / len(res)
print('accuracy on training data: ', accuracy)

res = np.argmax(Y, axis=1)
trueth = np.argmax(y_test, axis=1)
accuracy = np.sum(res == trueth) / len(res)
print('accuracy on testing data: ', accuracy)

plt.plot(range(len(L)), L)
plt.show()
