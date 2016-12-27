"""
Author: CHANDRAMOHAN T N
File: Process_data.py 
"""

import numpy as np
import nltk
from nltk.corpus import stopwords
s=set(stopwords.words('english'))
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

def get_reviews(i_f, o_f):
	f = open(i_f, 'r')
	g = open(o_f, 'w')
	txt = f.read()
	f.close()
	data = []
	for item in txt.split("</review_text>"):
		if '<review_text>' in item:
			d = item[item.find("<review_text>") + len("<review_text>"): ]
			d = nltk.word_tokenize(d)
			d = filter(lambda w: not w in s, d)
			d = filter(lambda w: len(w) > 2, d)
			a = []
			for i in d:
				if '-' in i:
					b = i.split('-')
					for j in b:
						a.append(j)
				else:
					a.append(i)
			d = filter(lambda w: w.isalpha(), a)
			a = ''
			for i in d:
				a = a + i.lower() + ' '
			a = a[0:-1]
			g.write(a + '\n')
	g.close()

def Get_vocab(if1, if2, o_f):
	vocab = set()
	f = open(if1, 'r')

	while 1:
		line = f.readline()
		line = line[0:-1]
		if len(line) > 0:
			items = line.split()
			for i in items:
				if i not in vocab:
					vocab.add(i)
		else:
			f.close()
			break

	f = open(if2, 'r')
	while 1:
		line = f.readline()
		line = line[0:-1]
		if len(line) > 0:
			items = line.split()
			for i in items:
				if i not in vocab:
					vocab.add(i)
		else:
			f.close()
			break
	vocab = list(vocab)
	np.savez_compressed(o_f, vocab=vocab)

def Get_vector(d, v):
	a = [0 for i in range(len(v))]
	for i in d:
		a[v[i]] += 1
	s = ''
	for i in a:
		s = s + str(i) + ','
	s = s[0:-1] 
	return s

def Create_dataset(if1, if2, of1, of2, of3, of4, v):
	vocab = {}
	for i in range(len(v)):
		vocab[v[i]] = i

	data = []
	labels = []
	f = open(if1, 'r')
	while 1:
		line = f.readline()
		line = line[0:-1]
		if len(line) > 0:
			items = line.split()
			data.append(items)
			labels.append(1)
		else:
			f.close()
			break

	f = open(if2, 'r')
	while 1:
		line = f.readline()
		line = line[0:-1]
		if len(line) > 0:
			items = line.split()
			data.append(items)
			labels.append(0)
		else:
			f.close()
			break

	idx = np.random.permutation(len(data))
	tr_idx = idx[0:int(len(data)*0.8)]
	te_idx = idx[int(len(data)*0.8):]

	f = open(of1, 'w')
	for i in tr_idx:
		d = Get_vector(data[i], vocab)
		f.write(d + '\n')
	f.close()
	f = open(of2, 'w')
	for i in te_idx:
		d = Get_vector(data[i], vocab)
		f.write(d + '\n')
	f.close()
	f = open(of3, 'w')
	for i in tr_idx:
		f.write(str(labels[i]) + '\n')
	f.close()
	f = open(of4, 'w')
	for i in te_idx:
		f.write(str(labels[i]) + '\n')
	f.close()
	print('Train & Test dataset created ....')

def main():
	'''
	i_f = '../Data/books/positive.review'
	o_f = '../Data/books/positive.dat'
	get_reviews(i_f, o_f)

	i_f = '../Data/dvd/positive.review'
	o_f = '../Data/dvd/positive.dat'
	get_reviews(i_f, o_f)

	i_f = '../Data/electronics/positive.review'
	o_f = '../Data/electronics/positive.dat'
	get_reviews(i_f, o_f)

	i_f = '../Data/kitchen_housewares/positive.review'
	o_f = '../Data/kitchen_housewares/positive.dat'
	get_reviews(i_f, o_f)

	i_f = '../Data/books/negative.review'
	o_f = '../Data/books/negative.dat'
	get_reviews(i_f, o_f)

	i_f = '../Data/dvd/negative.review'
	o_f = '../Data/dvd/negative.dat'
	get_reviews(i_f, o_f)

	i_f = '../Data/electronics/negative.review'
	o_f = '../Data/electronics/negative.dat'
	get_reviews(i_f, o_f)

	i_f = '../Data/kitchen_housewares/negative.review'
	o_f = '../Data/kitchen_housewares/negative.dat'
	get_reviews(i_f, o_f)
	'''
	if1 = '../Data/books/positive.dat'
	if2 = '../Data/books/negative.dat'
	o_f = '../Data/books/Vocab_books'
	Get_vocab(if1, if2, o_f)
	if1 = '../Data/electronics/positive.dat'
	if2 = '../Data/electronics/negative.dat'
	o_f = '../Data/electronics/Vocab_electronics'
	Get_vocab(if1, if2, o_f)
	if1 = '../Data/dvd/positive.dat'
	if2 = '../Data/dvd/negative.dat'
	o_f = '../Data/dvd/Vocab_dvd'
	Get_vocab(if1, if2, o_f)
	if1 = '../Data/kitchen_housewares/positive.dat'
	if2 = '../Data/kitchen_housewares/negative.dat'
	o_f = '../Data/kitchen_housewares/Vocab_kitchen_housewares'
	Get_vocab(if1, if2, o_f)

	if1 = '../Data/books/positive.dat'
	if2 = '../Data/books/negative.dat'
	of1 = '../Data/books/Train_x.csv'
	of2 = '../Data/books/Test_x.csv'
	of3 = '../Data/books/Train_y.csv'
	of4 = '../Data/books/Test_y.csv'
	d = np.load('../Data/books/Vocab_books.npz')
	v = d['vocab']
	Create_dataset(if1, if2, of1, of2, of3, of4, v)

	if1 = '../Data/dvd/positive.dat'
	if2 = '../Data/dvd/negative.dat'
	of1 = '../Data/dvd/Train_x.csv'
	of2 = '../Data/dvd/Test_x.csv'
	of3 = '../Data/dvd/Train_y.csv'
	of4 = '../Data/dvd/Test_y.csv'
	d = np.load('../Data/dvd/Vocab_dvd.npz')
	v = d['vocab']
	Create_dataset(if1, if2, of1, of2, of3, of4, v)

	if1 = '../Data/electronics/positive.dat'
	if2 = '../Data/electronics/negative.dat'
	of1 = '../Data/electronics/Train_x.csv'
	of2 = '../Data/electronics/Test_x.csv'
	of3 = '../Data/electronics/Train_y.csv'
	of4 = '../Data/electronics/Test_y.csv'
	d = np.load('../Data/electronics/Vocab_electronics.npz')
	v = d['vocab']
	Create_dataset(if1, if2, of1, of2, of3, of4, v)

	if1 = '../Data/kitchen_housewares/positive.dat'
	if2 = '../Data/kitchen_housewares/negative.dat'
	of1 = '../Data/kitchen_housewares/Train_x.csv'
	of2 = '../Data/kitchen_housewares/Test_x.csv'
	of3 = '../Data/kitchen_housewares/Train_y.csv'
	of4 = '../Data/kitchen_housewares/Test_y.csv'
	d = np.load('../Data/kitchen_housewares/Vocab_kitchen_housewares.npz')
	v = d['vocab']
	Create_dataset(if1, if2, of1, of2, of3, of4, v)

if __name__ == '__main__':
	main()

