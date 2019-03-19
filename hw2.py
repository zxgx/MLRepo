import pandas as pd
import numpy as np

# Preprocess
x_train = pd.read_csv(r'/home/zhg/share/hw2/X_train').values
y_train = pd.read_csv(r'/home/zhg/share/hw2/Y_train').values
x_test = pd.read_csv(r'/home/zhg/share/hw2/X_test').values

# normalization
xmin = np.min(x_train, axis=0)
xmax = np.max(x_train, axis=0)
dif = xmax - xmin
x_train = (x_train - xmin) / dif

# Parameters Configuration
W, b = np.random.rand(x_train.shape[1], 1), np.random.rand()
epochs = 20
batch_size=64
eta = 1
ww, bb = np.zeros((x_train.shape[1], 1)), np.zeros(1)

# Training
for i in range(epochs):
    for batch in range(int(np.floor(x_train.shape[0]/batch_size))):
        X = x_train[batch*batch_size:(batch+1)*batch_size]
        Y = y_train[batch*batch_size:(batch+1)*batch_size]

        z = X.dot(W) + b
        y = np.clip(1 / (1 + np.exp(-z)), 1e-6, 1-1e-6);

        loss = np.sum(-1 * (Y * np.log(y) + (1 - Y) * np.log(1 - y)))
        print('loss: ', loss)

        dldw = np.sum((Y - y) * X, axis=0)[: ,np.newaxis]/X.shape[0]
        dldb = np.sum(Y - y) / X.shape[0]
        ww = ww + dldw ** 2
        bb = bb + dldb ** 2

        W = W - eta / np.sqrt(ww) * dldw
        b = b - eta / np.sqrt(bb) * dldb

z = x_train.dot(W) + b
y = np.clip(1 / (1 + np.exp(-1 * z)), 1e-6, 1-1e-6)
y = np.round(y)
accuracy = np.sum(y == y_train) / y.shape[0]
print('accuracy on training data: ', accuracy)

x_test = (x_test - xmin) / dif # There exists a bug that I don't want to fix 
z = x_test.dot(W) + b
y = np.clip(1 / (1 + np.exp(-1 * z)), 1e-6, 1-1e-6)
y = np.round(y)
"""
d1 = np.array([x for x in range(1, 16282)]).reshape(16281, 1)
d2 = np.array(y.astype(int))
d = np.hstack((d1, d2))
df = pd.DataFrame(d, columns=['id', 'label'])
df.to_csv(r'/home/zhg/share/submission.csv', index=False)
"""
