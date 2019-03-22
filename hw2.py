import pandas as pd
import numpy as np

# Preprocess
x_train = pd.read_csv(r'/home/zhg/share/hw2/X_train').values
y_train = pd.read_csv(r'/home/zhg/share/hw2/Y_train').values
x_test = pd.read_csv(r'/home/zhg/share/hw2/X_test').values

# Normalization
xmin = np.min(x_train, axis=0)
xmax = np.max(x_train, axis=0)
dif = xmax - xmin
x_train = (x_train - xmin) / dif
x_test = (x_test - xmin) / dif

# Parameters Configuration
W, b = np.random.rand(x_train.shape[1], 1), np.random.rand()
epochs = 30
batch_size = 64
eta = 1
eps = 1e-8
ww, bb = np.zeros((x_train.shape[1], 1))+eps, np.zeros((1,))+eps

# Training
for i in range(epochs):
    for batch in range(int(np.ceil(x_train.shape[0]/batch_size))):
        X = x_train[batch*batch_size:(batch+1)*batch_size]
        Y = y_train[batch*batch_size:(batch+1)*batch_size]

        z = X.dot(W) + b
        y = 1 / (1 + np.exp(-z));
        
        ny = np.clip(y, eps, 1-eps)
        loss = np.mean(-1 * (Y * np.log(ny) + (1 - Y) * np.log(1 - ny)))
        print("loss: ", loss)

        dldw = np.sum((y - Y) * X, axis=0)[:, np.newaxis]
        dldb = np.sum(y - Y)
        
        ww = ww + dldw**2
        bb = bb + dldb**2

        W = W - eta / np.sqrt(ww) * dldw
        b = b - eta / np.sqrt(bb) * dldb

z = x_train.dot(W) + b
y = 1 / (1 + np.exp(-1 * z))
y = np.round(y)
accuracy = np.sum(y == y_train) / y.shape[0]
print('accuracy on training data: ', accuracy)

z = x_test.dot(W) + b
y = 1 / (1 + np.exp(-1 * z))
y = np.round(y)

d1 = np.array([x for x in range(1, 16282)]).reshape(16281, 1)
d2 = np.array(y.astype(int))
d = np.hstack((d1, d2))
df = pd.DataFrame(d, columns=['id', 'label'])
df.to_csv(r'/home/zhg/share/submission.csv', index=False)
