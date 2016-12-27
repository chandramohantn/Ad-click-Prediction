"""
Author: CHANDRAMOHAN T N
File: Single Value Structure.py 
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from LogisticRegression import *
from FTRL import *
from PerCoordinateLearningRates import *
from PoissonInclusion import *
from LogisticRegression import Log_reg
from FTRL import FTRL
from PerCoordinateLearningRates import PCLR
from PoissonInclusion import PI
from threading import Thread
from Queue import Queue
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

def plot(a, b, c, x, l):
	plt.plot(x, a, 'r', label='AUC Loss')
	plt.plot(x, b, 'g', label='Log Loss')
	plt.plot(x, c, 'b', label='Sq Error')
	plt.title(l)
	plt.legend(loc='upper right')
	plt.ylabel('Loss')
	plt.xlabel('Iteration')
	plt.savefig('Single Value Strcuture Test Loss.png')
	plt.close()

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

	l_reg = Log_reg(train_x, train_y)
	ftrl_reg = FTRL(train_x, train_y)
	pclr_reg = PCLR(train_x, train_y)
	pi_reg = PI(train_x, train_y)

	max_iter = 10000
	te_auc_loss = []
	te_log_loss = []
	te_sq_error = []
	for i in range(max_iter):))
		idx = np.random.randint(len(train_x))
		q1, q2, q3, q4 = Queue(), Queue(), Queue(), Queue()
		t1 = Thread(target=lambda q1, arg1, arg2: q1.put(l_reg.train_ogd(arg1, arg2)), args=(q1, 0.01, idx))
		t2 = Thread(target=lambda q2, arg1, arg2: q2.put(ftrl_reg.train_ogd(arg1, arg2)), args=(q2, 3, idx))
		t3 = Thread(target=lambda q3, arg1, arg2, arg3: q3.put(pclr_reg.train_ogd(arg1, arg2, arg3)), args=(q3, 0.01, 1.0, idx))
		t4 = Thread(target=lambda q4, arg1, arg2, arg3: q4.put(pi_reg.train_ogd(arg1, arg2, arg3)), args=(q4, 0.01, 40, idx))
		t1.start()
		t2.start()
		t3.start()
		t4.start()
		t1.join()
		t2.join()
		t3.join()
		t4.join()
		w = (q1.get() + q2.get() + q3.get() + q4.get()) * 1.0 / 4
		l_reg.update_weights(w)
		ftrl_reg.update_weights(w)
		pclr_reg.update_weights(w)
		pi_reg.update_weights(w)
		if i % 500 == 0:
			y_prob, y_pred = l_reg.predict(test_x)
			a, b, c = l_reg.evaluate(y_prob, test_y, test_x)
			te_auc_loss.append(a)
			te_log_loss.append(b)
			te_sq_error.append(c)

	r = [k for k in range(len(te_auc_loss))]
	plot(te_auc_loss, te_log_loss, te_sq_error, r, 'Single Value Strcuture Test_loss')
	np.savez_compressed('Model_single_value_structure', w=w)
	y_prob, y_pred = l_reg.predict(test_x)
	l_reg.evaluate(y_pred, test_y)

if __name__ == '__main__':
	main()
