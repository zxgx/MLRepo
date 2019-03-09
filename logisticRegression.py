from keras.datasets import mnist
import keras.utils as utils
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import utils as funcs
import matplotlib.pylab as plt

# Preprocess
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2])).astype('float') / 255
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2])).astype('float') / 255
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

iteration, batchs = 20, 100
eta = 0.0001
W = np.random.rand(x_train.shape[1], 10)
B = np.random.rand(1, 10)
L = []
for i in range(iteration):
    for batch in range(x_train.shape[0]//batchs):
        X = x_train[batch*batchs:(batch+1)*batchs, :]
        Z = X.dot(W) + B
        Y = funcs.softmax(Z)

        yh = y_train[batch*batchs:(batch+1)*batchs, :]
        loss = np.sum(-1 * yh * np.log(Y), axis=1)
        L.append(np.sum(loss))

        dldy = -1 * yh / Y
        dydz = Y * (1 - Y)
        dldz = dldy * dydz

        for col in range(W.shape[1]):
            dldw = np.sum(dldz[:, col][:, np.newaxis] * X, axis=0).T
            W[:, col] = W[:, col] - eta * dldw
        B = B - eta * np.sum(dldz, axis=0)

# validation on training & testing data
z = x_train.dot(W) + B
y = np.argmax(funcs.softmax(z), axis=1)
trueth = np.argmax(y_train, axis=1)

accuracy = np.sum(y == trueth)/len(y)
print('accuracy on training data', accuracy)

z = x_test.dot(W) + B
y = np.argmax(funcs.softmax(z), axis=1)
trueth = np.argmax(y_test, axis=1)
accuracy = np.sum(y == trueth)/len(y)
print('accuracy on testing data', accuracy)

# plt.plot(range(len(L)), L)
# plt.show()

# implemention with keras
# model = Sequential()
# model.add(Dense(units=10, activation='sigmoid', input_dim=28*28))
# model.add(Dense(units=10, activation='softmax'))
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='sgd',
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=20, batch_size=100)
#
# metrics = model.evaluate(x_train, y_train)
# print("Accuracy of training: ", metrics[1])
#
# metrics = model.evaluate(x_test, y_test)
# print("Accuracy of testing: ", metrics[1])
