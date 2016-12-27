"""
Author: CHANDRAMOHAN T N
File: Dropout.py 
"""

import numpy
import random
import math
import Plot
import decimal

class Neural_net:
    def __init__(self, i_nodes, h_nodes, o_nodes):
        self.i_nodes = i_nodes + 1
        self.h_nodes = h_nodes
        self.o_nodes = o_nodes

        wt = []
        for i in range(self.i_nodes):
            wt.append([0.0] * h_nodes)
        self.i_wt = wt
        wt = []
        for i in range(self.h_nodes):
            wt.append([0.0] * o_nodes)
        self.o_wt = wt
        for i in range(self.i_nodes):
            for j in range(self.h_nodes):
                self.i_wt[i][j] = random.uniform(-0.01, 0.01)
        for i in range(self.h_nodes):
            for j in range(self.o_nodes):
                self.o_wt[i][j] = random.uniform(-0.01, 0.01)

        self.i_act = [1.0] * self.i_nodes
        self.h_act = [1.0] * self.h_nodes
        self.o_act = [1.0] * self.o_nodes


    def update(self, inputs):
        for i in range(self.i_nodes - 1):
            self.i_act[i] = inputs[i]

        for i in range(self.h_nodes):
            tot = 0.0
            for j in range(self.i_nodes):
                tot += self.i_act[i] * self.i_wt[j][i]
            self.h_act[i] = self.activate_sigmoid(tot)

        for i in range(self.o_nodes):
            tot = 0.0
            for j in range(self.h_nodes):
                tot += self.h_act[i] * self.o_wt[j][i]
            self.o_act[i] = self.activate_sigmoid(tot)

        tot = sum(self.o_act)
        for i in range(self.o_nodes):
            self.o_act[i] = self.o_act[i] / tot

        return self.o_act


    def back_prop(self, targets, learn_ih, learn_ho, gamma):        
        self.o_delta = [0.0] * self.o_nodes
        for i in range(self.o_nodes):
            error = targets[i] - self.o_act[i]
            self.o_delta[i] = self.d_activate_sigmoid(self.o_act[i]) * error

        self.h_delta = [0.0] * self.h_nodes
        for i in range(self.h_nodes):
            error = 0.0
            for j in range(self.o_nodes):
                error += self.o_delta[j] * self.o_wt[i][j]
            self.h_delta[i] = self.d_activate_sigmoid(self.h_act[i]) * error

        for i in range(self.h_nodes):
            for j in range(self.o_nodes):
                step = (self.o_delta[j] * self.h_act[i]) + (2 * gamma * self.o_wt[i][j])
                self.o_wt[i][j] += (learn_ho * step)

        for i in range(self.i_nodes):
            for j in range(self.h_nodes):
                step = (self.h_delta[j] * self.i_act[i]) + (2 * gamma * self.i_wt[i][j])
                self.i_wt[i][j] += (learn_ih * step)

        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.o_act[k]) ** 2
        return error


    def reg_train(self, data_x, data_y, niter, learn_ih, learn_ho):
        gamma = 0.01
        while gamma < 100:
            print('Gamma: ' + str(gamma))
            err = []
            for i in range(niter):
                idx = numpy.random.permutation(len(data_x))
                print('Iteration: ' + str(i + 1))
                error = 0.0
                for j in idx:
                    inp = data_x[j]
                    outp = data_y[j]
                    self.update(inp)
                    error += self.back_prop(outp, learn_ih, learn_ho, gamma)
                err.append(error)
            Plot.Plot_error(err, niter, gamma)
            gamma += 10


    def train(self, data_x, data_y, niter, learn_ih, learn_ho):
        err = []
        gamma = 0
        for i in range(niter):
            idx = numpy.random.permutation(len(data_x))
            print('Iteration: ' + str(i + 1))
            error = 0.0
            for j in idx:
                inp = data_x[j]
                outp = data_y[j]
                self.update(inp)
                error += self.back_prop(outp, learn_ih, learn_ho, gamma)
            err.append(error)
        Plot.Plot_error(err, niter, gamma)
                

    def test(self, test_x):
        y_hat = []
        for i in test_x:
            out = self.update(i)
            y_hat.append(out.index(max(out)) + 1)
        return y_hat
                

    def print_weights(self):
        print('\n********* Input weights **************\n')
        for i in range(self.i_nodes):
            print(self.i_wt[i])
        print('\n')
        print('\n********* Output weights **************\n')
        for i in range(self.h_nodes):
            print(self.o_wt[i])
        print('\n')

    def activate_sigmoid(self, x):
        a = decimal.Decimal(-x).exp()
        b = decimal.Decimal(1.0)
        return float(b / (b + a))

    def d_activate_sigmoid(self, x):
        y = self.activate_sigmoid(x)
        return (y - y * y)

        



        
                
