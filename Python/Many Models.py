"""
Author: CHANDRAMOHAN T N
File: Many Models.py 
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threading import Thread
from Queue import Queue

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

def plot(a, b, c, x, l):
	plt.plot(x, a, 'r', label='AUC Loss')
	plt.plot(x, b, 'g', label='Log Loss')
	plt.plot(x, c, 'b', label='Sq Error')
	plt.title(l)
	plt.legend(loc='upper right')
	plt.ylabel('Loss')
	plt.xlabel('Iteration')
	plt.savefig('Many Models Test Loss.png')
	plt.close()

class Log_reg(object):

	def __init__(self, n):
		self.w = np.zeros((n, 1))

	def train_ogd(self, l_rate, x, y):
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

	def evaluate(self, y_pred, y_true, test_x):
		fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
		auc = metrics.auc(fpr, tpr)
		auc_loss = 1 - auc
		log_loss = log_likelihood(test_x, y_true, self.w)
		sq_error = np.sum((y_pred - y_true)**2) / y_pred.shape[0]
		return auc_loss, log_loss, sq_error

def main():
	i_f = '../Data/books/Train_x.csv'
	train_x = Get_data(i_f)
	train_x = np.concatenate((np.ones((len(train_x), 1)), np.array(train_x)), axis=1)
	i_f = '../Data/books/Test_x.csv'
	test_x = Get_data(i_f)
	test_x = np.concatenate((np.ones((len(test_x), 1)), np.array(test_x)), axis=1)
	i_f = '../Data/books/Train_y.csv'
	train_y = Get_labels(i_f)
	train_y = np.reshape(train_y, (train_x.shape[0], 1))
	i_f = '../Data/books/Test_y.csv'
	test_y = Get_labels(i_f)
	test_y = np.reshape(test_y, (test_x.shape[0], 1))

	l_reg1 = Log_reg(train_x.shape[1])
	l_reg2 = Log_reg(train_x.shape[1])
	l_reg3 = Log_reg(train_x.shape[1])
	max_iter = 20000
	te_auc_loss = []
	te_log_loss = []
	te_sq_error = []
	for i in range(max_iter):
		idx = np.random.randint(len(train_x))
		q1, q2, q3 = Queue(), Queue(), Queue()
		t1 = Thread(target=lambda q, arg1: q.put(l_reg1.train_ogd(args)), args=(q1, 0.01, idx))
		t2 = Thread(target=lambda q, arg1: q.put(l_reg2.train_ogd(args)), args=(q2, 0.02, idx))
		t3 = Thread(target=lambda q, arg1: q.put(l_reg3.train_ogd(args)), args=(q3, 0.03, idx))
		t1.start()
		t2.start()
		t3.start()
		l_reg1.update_weights(q1.get())
		l_reg2.update_weights(q2.get())
		l_reg3.update_weights(q3.get())
		if i % 500 == 0:
			print('Iteration: ' + str(i))
	t1.join()
	t2.join()
	t3.join()
	np.savez_compressed('Model_many_1', w=l_reg1.w)
	np.savez_compressed('Model_many_2', w=l_reg2.w)
	np.savez_compressed('Model_many_3', w=l_reg3.w)

if __name__ == '__main__':
	main()
