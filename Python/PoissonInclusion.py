"""
Author: CHANDRAMOHAN T N
File: Poisson Inclusion.py 
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from sklearn import metrics

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

def plot(a, b, c, x, l, n):
	plt.plot(x, a, 'r', label='AUC Loss')
	plt.plot(x, b, 'g', label='Log Loss')
	plt.plot(x, c, 'b', label='Sq Error')
	plt.title(l)
	plt.legend(loc='upper right')
	plt.ylabel('Loss')
	if n == 1:
		plt.xlabel('Iteration')
		plt.savefig('Poisson Inclusion Test Loss.png')
	else:
		plt.xlabel('P (probability limit)')
		plt.savefig('Poisson Inclusion Performance.png')
	plt.close()

class PI(object):

	def __init__(self, x, y):
		self.x = np.concatenate((np.ones((len(x), 1)), np.array(x)), axis=1)
		self.y = np.array(y).reshape(len(y), 1)
		self.w = np.zeros((self.x.shape[1], 1))
		self.count = np.zeros((1, self.w.shape[0]))

	def train_ogd(self, l_rate, p, idx):
		x = self.x[idx * 1: (idx+1) * 1]
		y = self.y[idx * 1: (idx+1) * 1]
		count = x.nonzero()[0]
		for i in count:
			self.count[i] += 1
		indx = ((np.transpose(self.count) > p).nonzero())[0]
		g = gradient(x, y, self.w)
		g_temp = np.zeros((self.w.shape[0], 1))
		for l in indx:
			g_temp[l] = g[l]
		self.w = self.w - l_rate * g_temp
		return self.w

	def update_weights(self, w):
		self.w = w

	def save_model(self):
		np.savez_compressed('Model_pi', w=self.w)

	def predict(self, x):
		labels = np.zeros((len(x), 1))
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
	i_f = '../Data/books/Test_x.csv'
	test_x = Get_data(i_f)
	test_x = np.concatenate((np.ones((len(test_x), 1)), np.array(test_x)), axis=1)
	i_f = '../Data/books/Train_y.csv'
	train_y = Get_labels(i_f)
	i_f = '../Data/books/Test_y.csv'
	test_y = Get_labels(i_f)
	test_y = np.reshape(test_y, (test_x.shape[0], 1))

	lr = 0.01
	max_iter = 25000
	p = [i for i in range(5, 51, 5)]
	prev_auc_loss = sys.maxint
	te_auc_loss = []
	te_log_loss = []
	te_sq_error = []
	for l in p:
		print('p: ' + str(l))
		pi_reg = PI(train_x, train_y)
		tr_auc_loss = []
		tr_log_loss = []
		tr_sq_error = []
		for i in range(max_iter):
			idx = np.random.randint(len(train_x))
			w = pi_reg.train_ogd(lr, l, idx)
			pi_reg.update_weights(w)
			if (i+1) % 500 == 0:
				y_prob, y_pred = pi_reg.predict(test_x)
				a, b, c = pi_reg.evaluate(y_prob, test_y, test_x)
				tr_auc_loss.append(a)
				tr_log_loss.append(b)
				tr_sq_error.append(c)
		y_prob, y_pred = pi_reg.predict(test_x)
		a, b, c = pi_reg.evaluate(y_prob, test_y, test_x)
		te_auc_loss.append(a)
		te_log_loss.append(b)
		te_sq_error.append(c)
		if a < prev_auc_loss:
			prev_auc_loss = a
			best_l = l
			pi_reg.save_model()
			r = [k for k in range(len(tr_auc_loss))]
			plot(tr_auc_loss, tr_log_loss, tr_sq_error, r, 'Poisson Inclusion Test_loss, p= ' + str(best_l), 1)
	plot(te_auc_loss, te_log_loss, te_sq_error, p, 'Poisson Inclusion Performance', 2)

if __name__ == '__main__':
	main()