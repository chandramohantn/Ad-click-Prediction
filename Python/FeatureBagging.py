"""
Author: CHANDRAMOHAN T N
File: Feature Bagging.py 
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def log_likelihood(x, y, w):
	p = sigmoid(np.dot(x, w))
	loss = -y * np.log(p) - (1 - y) * np.log(1 - p)
	return loss.mean()

def gradient(x, y, w):
	p = sigmoid(np.dot(x, w))
	grad = np.dot(np.transpose(x), (p - y))
	return grad.reshape(w.shape)

class Log_reg(object):

	def __init__(self, x, y):
		self.x = np.concatenate((np.ones((len(x), 1)), np.array(x)), axis=1)
		self.y = np.array(y).reshape(len(y), 1)
		self.w = np.zeros((self.x.shape[1], 1))

	def train_ogd(self, l_rate, idx):
		x = self.x[idx * 1: (idx+1) * 1]
		y = self.y[idx * 1: (idx+1) * 1]
		self.w = self.w - l_rate * gradient(x, y, self.w)
		return self.w

	def update_weights(self, w):
		self.w = w

	def save_model(self):
		np.savez_compressed('Model_l_reg', w=self.w)

	def predict(self, x):
		labels = [0 for i in xrange(len(x))]
		p = sigmoid(np.dot(x, self.w))
		for i in xrange(len(x)):
			if p[i] > 0.5:
				labels[i] = 1
		return p, labels

	def evaluate(self, y_pred, y_true):
		fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
		auc = metrics.auc(fpr, tpr)
		auc_loss = 1 - auc
		sq_error = np.sum((y_pred - y_true)**2) / y_pred.shape[0]
		return auc_loss, sq_error

def main():
	i_f = '../Data/books/Train_x.csv'
	train_x = Get_data(i_f)
	i_f = '../Data/books/Test_x.csv'
	test_x = Get_data(i_f)
	test_x = np.concatenate((np.ones((len(test_x), 1)), np.array(test_x)), axis=1)
	i_f = '../Data/books/Train_y.csv'
	train_y = Get_labels(i_f)
	i_f = '../Data/books/Test_y.csv'
	test_y = Get_labels(i_f)
	test_y = np.reshape(test_y, (test_x.shape[0], 1))

	ensemble_idx = []
	k = 5
	max_iter = 15000
	l = len(train_x[0])
	idx = np.random.permutation(l)
	b = int(l/k)
	for i in range(k-1):
		ensemble_idx.append(idx[i*b:((i+1)*b + b)])
	ensemble_idx.append(np.concatenate((idx[(k-1)*b:], idx[0:b])))

	l_reg1 = Log_reg(train_x[ensemble_idx[0]], train_y[ensemble_idx[0]])
	for i in range(max_iter):
		idx = np.random.randint(len(ensemble_idx[0]))
		w = l_reg1.train_ogd(0.01, idx)
		l_reg1.update_weights(w)

	l_reg2 = Log_reg(train_x[ensemble_idx[1]], train_y[ensemble_idx[1]])
	for i in range(max_iter):
		idx = np.random.randint(len(ensemble_idx[1]))
		w = l_reg2.train_ogd(0.01, idx)
		l_reg2.update_weights(w)

	l_reg3 = Log_reg(train_x[ensemble_idx[2]], train_y[ensemble_idx[2]])
	for i in range(max_iter):
		idx = np.random.randint(len(ensemble_idx[2]))
		w = l_reg3.train_ogd(0.01, idx)
		l_reg3.update_weights(w)

	l_reg4 = Log_reg(train_x[ensemble_idx[3]], train_y[ensemble_idx[3]])
	for i in range(max_iter):
		idx = np.random.randint(len(ensemble_idx[3]))
		w = l_reg4.train_ogd(0.01, idx)
		l_reg4.update_weights(w)

	l_reg5 = Log_reg(train_x[ensemble_idx[4]], train_y[ensemble_idx[4]])
	for i in range(max_iter):
		idx = np.random.randint(len(ensemble_idx[4]))
		w = l_reg5.train_ogd(0.01, idx)
		l_reg5.update_weights(w)

	np.savez_compressed('Model_bag1', w=l_reg1.w)
	np.savez_compressed('Model_bag2', w=l_reg2.w)
	np.savez_compressed('Model_bag3', w=l_reg3.w)
	np.savez_compressed('Model_bag4', w=l_reg4.w)
	np.savez_compressed('Model_bag5', w=l_reg5.w)

	y_prob1, y_pred1 = l_reg1.predict(test_x[ensemble_idx[0]])
	y_prob2, y_pred2 = l_reg2.predict(test_x[ensemble_idx[1]])
	y_prob3, y_pred3 = l_reg3.predict(test_x[ensemble_idx[2]])
	y_prob4, y_pred4 = l_reg4.predict(test_x[ensemble_idx[3]])
	y_prob5, y_pred5 = l_reg5.predict(test_x[ensemble_idx[4]])

	y_prob = np.concatenate((y_prob1, y_prob2))
	y_prob = np.concatenate((y_prob, y_prob3))
	y_prob = np.concatenate((y_prob, y_prob4))
	y_prob = np.concatenate((y_prob, y_prob5))
	y_prob = np.mean(y_prob, axis=1)

	y_pred = np.concatenate((y_pred1, y_pred2))
	y_pred = np.concatenate((y_pred, y_pred3))
	y_pred = np.concatenate((y_pred, y_pred4))
	y_pred = np.concatenate((y_pred, y_pred5))
	y_pred = np.sum(y_pred, axis=1)
	y = []
	for i in y_pred:
		if i > 2:
			y.append(1)
		else:
			y.append(0)
	y_pred = y
	a, b = l_reg1.evaluate(y_prob, test_y)
	print('AUC Loss: ' + str(a))
	print('Sq Error: ' + str(b))

if __name__ == '__main__':
	main()
