"""
features:
    Stochastic gradient descent
    Adagrad
    root mean square error evaluation
"""
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

# preprocessing
data = pd.read_csv(r'D:\BaiduNetdiskDownload\train.csv', encoding='big5')
# the factors taken into consideration 'NO', 'NO2', 'NOx', 'PM10', 'PM2.5', 'SO2', 'WIND_SPEED'
items = ['PM2.5']
data = data[data.iloc[:, 2].isin(items)].iloc[:, 2:]
dataDict = {}
for item in items:
    dataDict[item] = data[data.iloc[:, 0] == item].iloc[:, 1:].astype(float)
# trainData : validData = 4 : 1
trainData = [x * 20 + y for x in range(12) for y in range(20) if y not in [2, 7, 12, 17]]
validData = list(set(range(240)) - set(trainData))

# training parameters
prev, iteration = 6, 20
# eta = 0.00001
eta_b = np.ones(1) / 10
eta_w = np.ones((1, len(items)*prev)) / 10

W = np.random.rand(1, len(items)*prev)
b, averageError, L = np.random.rand(), 0, []
ww, bb = np.zeros((1, eta_w.shape[1])), np.zeros(1)
# training
for i in range(iteration):
    for day in trainData:
        for tick in range(prev, 24):
            X = dataDict[items[0]].iloc[day, tick - prev:tick]
            X = X[:, np.newaxis]
            for item in items[1:]:
                tmp = dataDict[item].iloc[day, tick - prev:tick]
                X = np.vstack((X, tmp[:, np.newaxis]))

            y = W.dot(X) + b

            error = (dataDict['PM2.5'].iat[day, tick] - y)
            L.append(abs(error))
            averageError = averageError + error ** 2

            dw = -2 * X.T * error
            db = -2 * error
            # stochastic gradient descent without adagrad
            # W = W - eta * dw
            # b = b - eta * db
            # stochastic gradient descent with adagrad
            ww = ww + dw ** 2
            bb = bb + db ** 2
            W = W - eta_w / np.sqrt(ww) * dw
            b = b - eta_b / np.sqrt(bb) * db
    # print('rmse: ', np.sqrt(averageError/(len(trainData)*(24-prev)*(i+1))))

print('The rms error of training: ', np.sqrt(averageError/(len(trainData)*(24-prev)*iteration)))
print('f(X) = b + W * X, in which b = %.2f, W = %s' % (b, W))
plt.figure()
plt.plot(range(len(trainData)*(24-prev)*iteration), np.array(L).flat)

# validation on training data
averageError = 0
for day in trainData:
    for tick in range(prev, 24):
        X = dataDict[items[0]].iloc[day, tick-prev:tick]
        X = X[:, np.newaxis]
        for item in items[1:]:
            tmp = dataDict[item].iloc[day, tick-prev:tick]
            X = np.vstack((X, tmp[:, np.newaxis]))

        y = W.dot(X) + b
        error = dataDict['PM2.5'].iat[day, tick] - y
        averageError = averageError + error ** 2
print('The rmse on training validation: ', np.sqrt(averageError/(len(trainData)*(24-prev))))

# validation
averageError = 0
for day in validData:
    for tick in range(prev, 24):
        X = dataDict[items[0]].iloc[day, tick-prev:tick]
        X = X[:, np.newaxis]
        for item in items[1:]:
            tmp = dataDict[item].iloc[day, tick-prev:tick]
            X = np.vstack((X, tmp[np.newaxis, :].T))

        y = W.dot(X) + b
        error = dataDict['PM2.5'].iat[day, tick] - y
        averageError += error ** 2

print('The rmse of validation ', np.sqrt(averageError/(len(validData)*(24-prev))))

# fully training
averageError = 0
for i in range(iteration):
    for day in range(data.shape[0]):
        for tick in range(prev, 20):
            X = dataDict[items[0]].iloc[day, tick - prev:tick]
            X = X[:, np.newaxis]
            for item in items[1:]:
                tmp = dataDict[item].iloc[day, tick - prev:tick]
                X = np.vstack((X, tmp[:, np.newaxis]))

            y = W.dot(X) + b

            error = (dataDict['PM2.5'].iat[day, tick] - y)
            averageError = averageError + error ** 2

            dw = -2 * X.T * error
            db = -2 * error
            # stochastic gradient descent with adagrad
            ww = ww + dw ** 2
            bb = bb + db ** 2
            W = W - eta_w / np.sqrt(ww) * dw
            b = b - eta_b / np.sqrt(bb) * db

print('rmse: ', np.sqrt(averageError/(len(trainData)*(24-prev)*iteration)))
plt.show()
