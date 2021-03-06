# Lab 11 MNIST and Convolutional Neural Network
import tensorflow as tf
import random
# import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
from abc import abstractmethod
import mytool
import matplotlib.pyplot as plt
from myplot import MyPlot
from mnist_cnn import MnistCNN


class XXX (MnistCNN):
    def init_network(self):
        self.set_placeholder(784, 10, 28, 28)

        # 1, 2
        CL_a = self.convolution_layer(self.X_2d, 3, 3, 1, 32, 1, 1)
        CL_a = self.relu(CL_a)
        CL_a_maxp = self.max_pool(CL_a, 2, 2, 2, 2)

        #3, 4
        CL_b = self.convolution_layer(CL_a_maxp, 3, 3, 32, 64, 1, 1)
        CL_b = self.relu(CL_b)
        CL_b_maxp = self.max_pool(CL_b, 2, 2, 2, 2)

        # 5
        reshaped = tf.reshape(CL_b_maxp, [-1, 7*7*64])
        hypo = self.fully_connected_layer(reshaped, 7*7*64, 10, 'input_l')
        self.set_hypothesis(hypo)

        self.set_cost_function()
        self.set_optimizer(0.001)


gildong = XXX()
gildong.learn_mnist(15, 100)
gildong.evaluate()
gildong.classify_random()
gildong.show_error()

'''
Epoch: 0001 cost = 0.347524221
Epoch: 0002 cost = 0.090191725
Epoch: 0003 cost = 0.064275292
Epoch: 0004 cost = 0.051426856
Epoch: 0005 cost = 0.042891532
Epoch: 0006 cost = 0.036567192
Epoch: 0007 cost = 0.032589161
Epoch: 0008 cost = 0.028781993
Epoch: 0009 cost = 0.024101281
Epoch: 0010 cost = 0.020956684
Epoch: 0011 cost = 0.019727421
Epoch: 0012 cost = 0.016530591
Epoch: 0013 cost = 0.014725357
Epoch: 0014 cost = 0.012840626
Epoch: 0015 cost = 0.012438118
Learning Finished!
Accuracy: 0.9888
Label:  [9]
Prediction:  [9]
'''
