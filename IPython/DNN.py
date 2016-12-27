from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
import sys

def Get_data(i_f, W, word_idx_map, vocab):
    f = open(i_f, 'r')
    data = []
    while 1:
        line = f.readline()
        line = line[0:-1]
        if len(line) > 0:
            items = map(int, line.split(','))
            idx = np.nonzero(items)[0]
            a = np.array([0 for i in range(300)])
            for i in idx:
                w = vocab[i]
                a = a + W[word_idx_map[w]]*items[i]
            data.append(a.tolist())
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

def load_bin_vec(fname, vocab):
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, 300)
    return word_vecs

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size, k), dtype='float32')            
    i = 0
    for word, vec in word_vecs.iteritems():
        W[i] = vec
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

v = np.load('../Data/books/Vocab_books.npz')
vocab = v['vocab']
w2v_file = '../Data/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin'
w2v = load_bin_vec(w2v_file, vocab)
print('Word2Vec loaded ....')
print("num words already in word2vec: " + str(len(w2v)))
w2v = add_unknown_words(w2v, vocab)

W, word_idx_map = get_W(w2v)
i_f = '../Data/books/Train_x.csv'
train_x = Get_data(i_f, W, word_idx_map, vocab)
train_x = np.concatenate((np.ones((len(train_x), 1)), np.array(train_x)), axis=1)
i_f = '../Data/books/Test_x.csv'
test_x = Get_data(i_f, W, word_idx_map, vocab)
test_x = np.concatenate((np.ones((len(test_x), 1)), np.array(test_x)), axis=1)
i_f = '../Data/books/Train_y.csv'
train_y = Get_labels(i_f)
train_y = np.reshape(train_y, (train_x.shape[0], 1))
i_f = '../Data/books/Test_y.csv'
test_y = Get_labels(i_f)
test_y = np.reshape(test_y, (test_x.shape[0], 1))

n = 301
print('Build model...')
model = Sequential()
model.add(Dense(100, input_dim=n, init='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

l = len(train_x)
print('Training .....')
for i in range(15000):
    idx = np.random.randint(l)
    a = np.reshape(train_x[idx], (1, 301))
    b = np.reshape(train_y[idx], (1, 1))
    model.train_on_batch(a, b)
model.save_weights('Model_DNN', overwrite=True)

print("Testing...")
#model.load_weights('Model_DNN')
y_pred = model.predict_classes(test_x)
y_prob = model.predict_proba(test_x)

print('Accuracy:')
print(accuracy_score(test_y, y_pred))
print('Precison : Recall : F-score')
print(precision_recall_fscore_support(test_y, y_pred))
