import numpy as np

def load_train_data():
	data_type = np.float64
	H, W = 48, 48
	num_train, num_val = 24000, 4709

	train = np.genfromtxt(r'data\ml2019spring-hw3\train.csv', delimiter=',', dtype='str', skip_header=1)
	X = []
	y = []

	for i in range(train.shape[0]):
		X.append(np.array(train[i,1].split(' ')).reshape(1, 48, 48))
		y.append(train[i,0])
	X = np.array(X, dtype=data_type)
	y = np.array(y, dtype=int)
	
	shuffle_mask = np.random.shuffle(np.arange(X.shape[0]))
	X[:] = X[shuffle_mask]
	y[:] = y[shuffle_mask]
	
	X_train = X[num_val:num_val+num_train]/255
	y_train = y[num_val:num_val+num_train]
	X_val = X[:num_val]/255
	y_val = y[:num_val]

	return (X_train, y_train, X_val, y_val)
	
def load_test_data():
	data_type = np.float64
	H, W = 48, 48
	
	tmp = np.genfromtxt(r'data\ml2019spring-hw3\test.csv', delimiter=',', dtype='str', skip_header=1)
	X = []
	for i in range(tmp.shape[0]):
		X.append(np.array(tmp[i,1].split(' '), dtype=data_type).reshape(1, 48, 48))
	test_set = np.array(X)/255
	
	return test_set

