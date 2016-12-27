"""
Author: CHANDRAMOHAN T N
File: Feature Scaling.py 
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import preprocessing
import LogisticRegression
from LogisticRegression import Log_reg

def Get_data(i_f):
	f = open(i_f, 'r')
	data = []
	while 1:
		line = f.readline()
		line = line[0:-1]
		if len(line) > 0:
			items = map(int, line.split(','))
			data.append(items)
		else:
			f.close()
			break
	return data

def Get_labels(i_f):
	f = open(i_f, 'r')
	labels = []
	while 1:
		line = f.readline()
		line = line[0:-1]
		if len(line) > 0:
			labels.append(int(line))
		else:
			f.close
			break
	return labels

def main():
	i_f = '../Data/books/Train_x.csv'
	train_x = Get_data(i_f)
	i_f = '../Data/books/Test_x.csv'
	test_x = Get_data(i_f)
	i_f = '../Data/books/Train_y.csv'
	train_y = Get_labels(i_f)
	i_f = '../Data/books/Test_y.csv'
	test_y = Get_labels(i_f)
	test_y = np.reshape(test_y, (len(test_x), 1))
	acc = []

	# Normalization
	normalizer = preprocessing.Normalizer().fit(train_x)
	x_train = normalizer.transform(train_x)
	x_test = normalizer.transform(test_x)
	x_test = np.concatenate((np.ones((len(test_x), 1)), np.array(x_test)), axis=1)
	l_reg = Log_reg(x_train, train_y)
	for i in range(15000):
		idx = np.random.randint(len(train_x))
		l_reg.train_ogd(0.01, idx)
	np.savez_compressed('Model_normalization', w=l_reg.w)
	y_prob, y_pred = l_reg.predict(x_test)
	a, b, c = l_reg.evaluate(y_prob, test_y, x_test)
	print('After Normalization')
	print('AUC Loss: ' + str(a))
	print('Log Loss: ' + str(b))
	print('Sq Loss: ' + str(c))

	# Min-Max
	min_max_scaler = preprocessing.MinMaxScaler().fit(train_x)
	x_train = min_max_scaler.transform(train_x)
	x_test = min_max_scaler.transform(test_x)
	x_test = np.concatenate((np.ones((len(test_x), 1)), np.array(x_test)), axis=1)
	l_reg = Log_reg(x_train, train_y)
	for i in range(15000):
		idx = np.random.randint(len(train_x))
		l_reg.train_ogd(0.01, idx)
	np.savez_compressed('Model_min_max', w=l_reg.w)
	y_prob, y_pred = l_reg.predict(x_test)
	a, b, c = l_reg.evaluate(y_prob, test_y, x_test)
	print('After Min-Max Scaling')
	print('AUC Loss: ' + str(a))
	print('Log Loss: ' + str(b))
	print('Sq Loss: ' + str(c))

	# Standardization
	scaler = preprocessing.StandardScaler().fit(train_x)
	x_train = scaler.transform(train_x)
	x_test = scaler.transform(test_x)
	x_test = np.concatenate((np.ones((len(test_x), 1)), np.array(x_test)), axis=1)
	l_reg = Log_reg(x_train, train_y)
	for i in range(15000):
		idx = np.random.randint(len(train_x))
		l_reg.train_ogd(0.01, idx)
	np.savez_compressed('Model_standardization', w=l_reg.w)
	y_prob, y_pred = l_reg.predict(x_test)
	a, b, c = l_reg.evaluate(y_prob, test_y, x_test)
	print('After Standardization')
	print('AUC Loss: ' + str(a))
	print('Log Loss: ' + str(b))
	print('Sq Loss: ' + str(c))

if __name__ == '__main__':
	main()
