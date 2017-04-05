# Lab 6 Softmax Classifier
import tensorflow as tf
import numpy as np
from neural_network import NeuralNetwork
from file2buffer import File2Buffer
import numpy as np
from mytype import MyType
from softmax_onehot import SoftmaxOnehot


class XXX (SoftmaxOnehot):
    def init_network(self):
        self.set_placeholder(16, 1)
        self.set_one_hot(7)

        logit = self.create_layer(self.X, 16, 7)

        self.set_hypothesis(logit)
        self.set_cost_function(logit)
        self.set_optimizer(MyType.GRADIENTDESCENT, 0.1)

gildong = XXX()
gildong.learn_with_file('data-04-zoo.csv', 2000, 100)
gildong.evaluate('data-04-zoo.csv')
gildong.show_error()

'''
Step:  1800	Loss: 0.060	Acc: 100.00%
Step:  1900	Loss: 0.057	Acc: 100.00%
Step:  2000	Loss: 0.054	Acc: 100.00%
'''

