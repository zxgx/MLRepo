import pandas as pd
import numpy as np

# Preprocess
x_train = pd.read_csv(r'/home/zhg/share/hw2/X_train').values
y_train = pd.read_csv(r'/home/zhg/share/hw2/Y_train').values
xmin = np.min(x_train, axis=0)
xmax = np.max(x_train, axis=0)
dif = xmax - xmin
x_train = (x_train - xmin) / dif

# Parameters Configuration
W, b = np.random.rand(x_train.shape[1], 1), np.random.rand()
iteration = 20
eta = 1
ww, bb = np.zeros((x_train.shape[1], 1)), np.zeros(1)
# Loss = []
# Training
for i in range(iteration):
    z = x_train.dot(W) + b
    y = 1 / (1 + np.exp(-1 * z))
    
    loss = np.sum(-1 * y_train * np.log(y))
    print('loss: ', loss)
    # Loss.append(loss)
    
    dldy = -1 * y_train / y
    dydz = y * (1 - y)
    dldw = np.sum(dldy * dydz * x_train, axis=0).T
    dldb = dldy * dydz
    ww = ww + dldw ** 2
    bb = bb + dldb ** 2
    print('dldw: ', dldw)
    print('dldb: ', dldb)

    W = W - eta / np.sqrt(ww) * dldw
    b = b - eta / np.sqrt(bb) * dldb

z = x_train.dot(W) + b
y = 1 / (1 + np.exp(-1 * z))
y = np.clip(y, 0, 1)
print(y.shape)
accuracy = np.sum(y == y_train) / y.shape[0]
print('accuracy: ', accuracy)

