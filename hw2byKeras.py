from keras.models import Sequential
from keras.layers import Dense
import keras.utils as utils
import numpy as np
import pandas as pd

# Preprocess
x_train = pd.read_csv(r'/home/zhg/share/hw2/X_train').values
y_train = pd.read_csv(r'/home/zhg/share/hw2/Y_train').values
x_test = pd.read_csv(r'/home/zhg/share/hw2/X_test').values

# normalization
xmin = np.min(x_train, axis=0)
xmax = np.max(x_train, axis=0)
dif = xmax - xmin
x_train = (x_train - xmin) / dif
x_test = (x_test - xmin) / dif

model = Sequential()

model.add(Dense(units=100, activation='relu', input_dim=x_train.shape[1]))
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])

model.fit(x_train, y_train,epochs=50, batch_size=128)

metrics=model.evaluate(x_train, y_train)
print("loss: ", metrics[0])
print("accuracy: ", metrics[1])

res = model.predict(x_test, batch_size=32)
res = np.round(res)

d1 = np.array([x for x in range(1, 16282)]).reshape(16281, 1)
d2 = np.array(res.astype(int))
d = np.hstack((d1, d2))
df = pd.DataFrame(d, columns=['id', 'label'])
df.to_csv(r'/home/zhg/share/submission.csv', index=False)

