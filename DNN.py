from keras.datasets import mnist
import keras.utils as utils
import numpy as np

def softmax(z):
    z = np.exp(z)
    d = np.sum(z, axis=1, keepdims=True)
    return z/d

# Preprocess
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2])).astype(np.float64) / 255
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2])).astype(np.float64) / 255
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

# Network construction
neurons = (x_train.shape[1], 500, 500, 10)
layers = len(neurons)-1
eta = 0.01
eps = 1e-8
W, B, L = [], [], []
ww, bb = [], []
for i in range(layers):
    W.append(np.random.rand(neurons[i], neurons[i+1]))
    B.append(np.random.rand(1, neurons[i+1]))
    ww.append(np.zeros((neurons[i], neurons[i+1]))+eps)
    bb.append(np.zeros((1, neurons[i+1]))+eps)

# Training
epochs, batch_size = 20, 64
for i in range(epochs):
    print("epoch: ", i)
    for batch in range(int(np.ceil(x_train.shape[0]/batch_size))):
        X = x_train[batch*batch_size : batch_size*(batch+1)]
        a = []

        # forward pass
        for layer in range(layers):
            z = X.dot(W[layer]) + B[layer]

            if layer != layers-1:
                a.append(1/(1+np.exp(-z)))
            else:
                a.append(softmax(z))
            X = a[-1]
        X = np.clip(X, eps, 1-eps)
        # recording error, cross entropy
        Y = y_train[batch*batch_size : (batch+1)*batch_size]
        ln = np.mean(np.sum(-1 * np.log(X) * Y, axis=1))
        # print("loss: ", ln)
        L.append(ln)

        # backward pass
        plpa = -Y / X
        for layer in reversed(range(layers)):
            papz = a[layer] * (1 - a[layer])    # sigmoid or softmax derivation
            if layer == 0:
                pzpw = x_train[batch*batch_size:(batch+1)*batch_size]
            else:
                pzpw = a[layer-1]
            plpz = plpa * papz

            plpa = plpz.dot(W[layer].T)
            
            gradw = pzpw.T.dot(plpz)
            ww[layer]=ww[layer]+gradw**2
            W[layer]=W[layer]-eta*gradw/np.sqrt(ww[layer])
            gradb = np.sum(plpz, axis=0, keepdims=True)
            bb[layer]=bb[layer]+gradb**2
            B[layer]=B[layer]-eta*gradb/np.sqrt(bb[layer])

# validation on training data
X = x_train
Y = x_test
for layer in range(layers):
    X = X.dot(W[layer])+B[layer]
    Y = Y.dot(W[layer])+B[layer]
    if layer == layers-1:
        X = softmax(X)
        Y = softmax(Y)
    else:
        X = 1/(1+np.exp(-X))
        Y = 1/(1+np.exp(-Y))

res = np.argmax(X, axis=1)
trueth = np.argmax(y_train, axis=1)
accuracy = np.sum(res == trueth) / len(res)
print('accuracy on training data: ', accuracy)

res = np.argmax(Y, axis=1)
trueth = np.argmax(y_test, axis=1)
accuracy = np.sum(res == trueth) / len(res)
print('accuracy on testing data: ', accuracy)
